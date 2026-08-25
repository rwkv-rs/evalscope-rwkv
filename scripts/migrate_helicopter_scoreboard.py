"""Migrate the legacy Helicopter scoreboard into Scoreboard contract-v4 storage.

The command is intentionally dry-run by default.  It reads database credentials
from environment variables and requires an explicit model-to-weight mapping so
that a model name is never mistaken for a real weight content digest.
"""

from __future__ import annotations

import argparse
import asyncio
import asyncpg
import json
import math
import os
import re
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from typing import Any, Iterable, Literal, Mapping, Sequence

EvaluationSource = Literal['lighteval', 'evalscope', 'lm-eval-harness']
WkvMode = Literal['fp16', 'fp32io16']


class StrictModel(BaseModel):
    """Reject unknown fields and non-standard numeric values."""

    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, strict=True)


def _require_trimmed(value: str, name: str) -> None:
    if not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")


def _sorted_unique(values: list[str], name: str) -> list[str]:
    if any(not value.strip() or value != value.strip() for value in values):
        raise ValueError(f"{name} must contain unique non-empty trimmed strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique non-empty trimmed strings")
    return sorted(values)


class Task(StrictModel):
    """Scoreboard-v1 task identity and display metadata."""

    identity: str = Field(min_length=1, max_length=1000)
    weight_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    weight_display_name: str = Field(min_length=1, max_length=500)
    wkv_mode: WkvMode
    benchmark: str = Field(min_length=1, max_length=500)
    task_name: str = Field(min_length=1, max_length=500)
    task_version: str = Field(min_length=1, max_length=100)
    dataset: str | None = Field(default=None, min_length=1, max_length=500)
    subset: str | None = Field(default=None, min_length=1, max_length=500)
    evaluation_splits: list[str]
    languages: list[str]
    tags: list[str]

    @model_validator(mode='after')
    def validate_task(self) -> 'Task':
        for name in (
            'identity',
            'weight_display_name',
            'benchmark',
            'task_name',
            'task_version',
        ):
            _require_trimmed(getattr(self, name), name)
        for name in ('dataset', 'subset'):
            value = getattr(self, name)
            if value is not None:
                _require_trimmed(value, name)
        expected = f"{self.weight_sha256}:{self.wkv_mode}:{self.task_name}"
        if self.identity != expected:
            raise ValueError('task identity does not match weight, WKV mode, and task')
        self.evaluation_splits = _sorted_unique(self.evaluation_splits, 'evaluation_splits')
        self.languages = _sorted_unique(self.languages, 'languages')
        self.tags = _sorted_unique(self.tags, 'tags')
        return self


def canonical_json(value: Any) -> bytes:
    """Encode a value using the Scoreboard canonical JSON representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')


def json_sha256(value: Any) -> str:
    """Hash a value after canonical JSON encoding."""

    import hashlib

    return hashlib.sha256(canonical_json(value)).hexdigest()


def campaign_run_key(campaign: Any) -> str:
    """Calculate the order-independent campaign contract key."""

    if isinstance(campaign, BaseModel):
        payload = campaign.model_dump(mode='json')
    else:
        payload = dict(campaign)
    payload.pop('run_key', None)
    for name in (
        'configured_benchmarks',
        'resolved_benchmarks',
        'skipped_benchmarks',
    ):
        benchmarks = payload.get(name)
        if isinstance(benchmarks, list):
            payload[name] = sorted(benchmarks)
    expected_tasks = payload.get('expected_tasks')
    if isinstance(expected_tasks, list):
        tasks: list[dict[str, Any]] = []
        for value in expected_tasks:
            if isinstance(value, BaseModel):
                task = value.model_dump(mode='json')
            else:
                task = dict(value)
            for name in ('evaluation_splits', 'languages', 'tags'):
                values = task.get(name)
                if isinstance(values, list):
                    task[name] = sorted(values)
            tasks.append(task)
        payload['expected_tasks'] = sorted(
            tasks,
            key=lambda task: (str(task.get('identity', '')), canonical_json(task)),
        )
    return json_sha256(payload)


class CampaignCreate(StrictModel):
    """Scoreboard-v1 campaign contract used to generate target rows."""

    schema_version: Literal['scoreboard-v1']
    run_key: str = Field(pattern=r'^[0-9a-f]{64}$')
    source: EvaluationSource
    config_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    registry_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    contract_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    configured_benchmarks: list[str] = Field(min_length=1)
    resolved_benchmarks: list[str]
    skipped_benchmarks: list[str]
    expected_tasks: list[Task]
    rerun_reason: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode='after')
    def validate_campaign(self) -> 'CampaignCreate':
        for name in (
            'configured_benchmarks',
            'resolved_benchmarks',
            'skipped_benchmarks',
        ):
            setattr(self, name, _sorted_unique(getattr(self, name), name))
        if not set(self.resolved_benchmarks).isdisjoint(self.skipped_benchmarks):
            raise ValueError('resolved and skipped benchmarks must be disjoint')
        if set(self.resolved_benchmarks) | set(self.skipped_benchmarks) != set(self.configured_benchmarks):
            raise ValueError('benchmark status must partition configured benchmarks')
        identities = [task.identity for task in self.expected_tasks]
        if len(identities) != len(set(identities)):
            raise ValueError('expected task identities must be unique')
        benchmarks = {task.benchmark for task in self.expected_tasks}
        if benchmarks != set(self.resolved_benchmarks):
            raise ValueError('resolved benchmarks must match expected tasks')
        if self.rerun_reason is not None:
            _require_trimmed(self.rerun_reason, 'rerun_reason')
        self.expected_tasks = sorted(
            self.expected_tasks,
            key=lambda task: (task.identity, canonical_json(task.model_dump(mode='json'))),
        )
        expected = campaign_run_key(self)
        if self.run_key != expected:
            raise ValueError(f"run_key does not match normalized campaign payload; expected {expected}")
        return self


class Sample(StrictModel):
    """One normalized Scoreboard sample containing all legacy attempts."""

    sample_index: int = Field(ge=0)
    document_index: int = Field(ge=0)
    document: dict[str, JsonValue]
    metrics: dict[str, JsonValue]
    model_response: dict[str, JsonValue]


class TaskPublication(StrictModel):
    """Scoreboard-v1 task payload used for validation and content hashing."""

    schema_version: Literal['scoreboard-v1']
    campaign_id: str = Field(min_length=1, max_length=100)
    task: Task
    result_files: list[dict[str, JsonValue]]
    task_config: dict[str, JsonValue]
    environment: dict[str, JsonValue]
    sampling_config: dict[str, JsonValue]
    primary_metric: str = Field(min_length=1, max_length=300)
    metrics: dict[str, float]
    diagnostics: dict[str, JsonValue]
    samples: list[Sample]

    @model_validator(mode='after')
    def validate_publication(self) -> 'TaskPublication':
        try:
            uuid.UUID(self.campaign_id)
        except ValueError as error:
            raise ValueError('campaign_id must be a UUID') from error
        _require_trimmed(self.primary_metric, 'primary_metric')
        if not self.metrics or any(not name.strip() or name != name.strip() for name in self.metrics):
            raise ValueError('metrics must use non-empty trimmed metric names')
        if self.primary_metric not in self.metrics:
            raise ValueError('primary_metric must exist in metrics')
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError('metrics must be finite')
        indexes = [sample.sample_index for sample in self.samples]
        if indexes != list(range(len(self.samples))):
            raise ValueError('sample_index values must be consecutive from zero')
        return self


async def _configure_connection(connection: asyncpg.Connection) -> None:
    for type_name in ('json', 'jsonb'):
        await connection.set_type_codec(
            type_name,
            schema='pg_catalog',
            encoder=lambda value: json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(',', ':'),
            ),
            decoder=json.loads,
            format='text',
        )


MIGRATION_NAME = 'helicopter-scoreboard-contract-v4'
MIGRATION_VERSION = 1
PUBLISHER = 'legacy-helicopter-migration-v1'
NAMESPACE = uuid.UUID('9be17da4-60ef-52d3-a691-0c19c63a4991')
SHA256_PATTERN = re.compile(r'^[0-9a-f]{64}$')
EVALUATOR_SOURCES = {
    'lighteval': 'lighteval',
    'function_calling': 'evalscope',
}
MIGRATION_CONTRACT = {
    'name': MIGRATION_NAME,
    'version': MIGRATION_VERSION,
    'campaigns': {
        'latest': 'one snapshot per source',
        'history': 'one campaign per superseded score',
    },
    'source_mapping': EVALUATOR_SOURCES,
    'samples': 'group completions by legacy task and sample index',
    'metrics': 'finite scalar aggregates; full legacy value in diagnostics',
}
CONTRACT_SHA256 = json_sha256(MIGRATION_CONTRACT)

LEGACY_REQUIRED_COLUMNS = {
    'model': {
        'model_id',
        'data_version',
        'arch_version',
        'num_params',
        'model_name',
    },
    'benchmark': {
        'benchmark_id',
        'benchmark_name',
        'benchmark_split',
        'url',
        'status',
        'num_samples',
    },
    'benchmark_catalog': {
        'catalog_id',
        'benchmark_name',
        'benchmark_split',
        'field',
        'source',
        'source_family',
        'target_kind',
        'run_status',
        'scope',
        'metadata',
        'created_at',
        'updated_at',
    },
    'task': {
        'task_id',
        'config_path',
        'evaluator',
        'is_param_search',
        'is_tmp',
        'created_at',
        'status',
        'git_hash',
        'model_id',
        'benchmark_id',
        'desc',
        'sampling_config',
        'log_path',
    },
    'scores': {'score_id', 'task_id', 'cot_mode', 'metrics', 'created_at'},
    'completions': {
        'completions_id',
        'task_id',
        'context',
        'sample_index',
        'avg_repeat_index',
        'pass_index',
        'created_at',
        'status',
    },
    'eval': {
        'eval_id',
        'completions_id',
        'answer',
        'ref_answer',
        'is_passed',
        'fail_reason',
        'created_at',
    },
    'checker': {
        'checker_id',
        'completions_id',
        'answer_correct',
        'instruction_following_error',
        'world_knowledge_error',
        'math_error',
        'reasoning_logic_error',
        'thought_contains_correct_answer',
        'needs_human_review',
        'reason',
        'created_at',
    },
}


class MigrationError(RuntimeError):
    """Base class for a migration that cannot safely proceed."""


class MigrationConflictError(MigrationError):
    """The target contains data that differs from the deterministic plan."""


@dataclass(frozen=True, slots=True)
class ModelMapping:
    weight_sha256: str
    weight_display_name: str
    wkv_mode: str
    wkv_mode_source: str


@dataclass(frozen=True, slots=True)
class TaskPlan:
    evaluation_id: uuid.UUID
    source_score_id: int
    created_at: datetime
    content_sha256: str
    publication: TaskPublication


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    campaign_id: uuid.UUID
    campaign: CampaignCreate
    created_at: datetime
    completed_at: datetime
    tasks: tuple[TaskPlan, ...]
    kind: str


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    campaigns: tuple[CampaignPlan, ...]
    source_task_count: int
    selected_score_count: int
    excluded_task_count: int

    @property
    def task_count(self) -> int:
        return sum(len(campaign.tasks) for campaign in self.campaigns)

    @property
    def sample_count(self) -> int:
        return sum(len(task.publication.samples) for campaign in self.campaigns for task in campaign.tasks)

    @property
    def attempt_count(self) -> int:
        return sum(
            len(sample.model_response.get('attempts', []))
            for campaign in self.campaigns
            for task in campaign.tasks
            for sample in task.publication.samples
        )


@dataclass(frozen=True, slots=True)
class MigrationSummary:
    mode: str
    source_task_count: int
    selected_score_count: int
    excluded_task_count: int
    planned_campaign_count: int
    planned_task_count: int
    planned_sample_count: int
    planned_attempt_count: int
    snapshot_campaign_count: int
    historical_campaign_count: int
    campaigns_to_create: int
    campaigns_reused: int
    tasks_to_create: int
    tasks_reused: int


def _trimmed(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MigrationError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise MigrationError(f"{field} must not contain surrounding whitespace")
    return value


def parse_model_mappings(value: Any) -> dict[str, ModelMapping]:
    """Validate the explicit legacy model mapping document."""

    if not isinstance(value, dict) or not value:
        raise MigrationError('model mapping must be a non-empty JSON object')
    result: dict[str, ModelMapping] = {}
    for raw_name, raw_mapping in value.items():
        model_name = _trimmed(raw_name, field='model name')
        if not isinstance(raw_mapping, dict):
            raise MigrationError(f"mapping for {model_name!r} must be an object")
        allowed = {
            'weight_sha256',
            'weight_display_name',
            'wkv_mode',
            'wkv_mode_source',
        }
        unexpected = sorted(set(raw_mapping) - allowed)
        if unexpected:
            raise MigrationError(f"mapping for {model_name!r} has unsupported fields: " + ', '.join(unexpected))
        weight_sha256 = raw_mapping.get('weight_sha256')
        if not isinstance(weight_sha256, str) or not SHA256_PATTERN.fullmatch(weight_sha256):
            raise MigrationError(f"mapping for {model_name!r} requires a lowercase 64-hex "
                                 'weight_sha256')
        display_name = raw_mapping.get('weight_display_name', model_name)
        display_name = _trimmed(
            display_name,
            field=f"weight_display_name for {model_name!r}",
        )
        if len(display_name) > 500:
            raise MigrationError(f"weight_display_name for {model_name!r} exceeds 500 characters")
        wkv_mode = raw_mapping.get('wkv_mode')
        if wkv_mode not in {'fp16', 'fp32io16'}:
            raise MigrationError(f"mapping for {model_name!r} requires wkv_mode fp16 or fp32io16")
        wkv_mode_source = raw_mapping.get('wkv_mode_source', 'explicit-model-map')
        wkv_mode_source = _trimmed(
            wkv_mode_source,
            field=f"wkv_mode_source for {model_name!r}",
        )
        result[model_name] = ModelMapping(
            weight_sha256=weight_sha256,
            weight_display_name=display_name,
            wkv_mode=wkv_mode,
            wkv_mode_source=wkv_mode_source,
        )
    return result


def load_model_mappings(path: Path) -> dict[str, ModelMapping]:
    """Load and validate a model mapping from a local JSON file."""

    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except OSError as error:
        raise MigrationError(f"cannot read model mapping {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise MigrationError(f"model mapping is not valid JSON: {error}") from error
    return parse_model_mappings(raw)


def _json_value(value: Any) -> Any:
    """Return strict JSON while retaining non-standard source values as text."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {'legacy_non_finite_number': str(value)}
    if isinstance(value, Decimal):
        if value.is_finite():
            return float(value)
        return {'legacy_non_finite_number': str(value)}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return {'legacy_bytes_hex': value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return {'legacy_python_value': str(value), 'type': type(value).__name__}


def split_scalar_metrics(value: Any) -> tuple[dict[str, float], dict[str, Any]]:
    """Promote finite numeric aggregates and preserve the full legacy value."""

    normalized = _json_value(value)
    if not isinstance(value, Mapping):
        return {}, {'legacy_metrics': normalized}
    scalars: dict[str, float] = {}
    for raw_name, metric_value in value.items():
        name = str(raw_name)
        if not name.strip() or name != name.strip() or isinstance(metric_value, bool):
            continue
        if isinstance(metric_value, (int, float, Decimal)):
            numeric = float(metric_value)
            if math.isfinite(numeric):
                scalars[name] = numeric
    return scalars, {'legacy_metrics': normalized}


def choose_primary_metric(metrics: Mapping[str, float]) -> str:
    """Choose a deterministic primary metric without recomputing scores."""

    if not metrics:
        raise MigrationError('legacy score has no finite scalar metric')
    candidates = [name for name in metrics if not name.lower().endswith(('_stderr', '_std', '_variance'))]
    if not candidates:
        candidates = list(metrics)
    exact_priority = (
        'accuracy',
        'exact_match',
        'acc',
        'em',
        'score',
        'error_rate',
    )
    for name in exact_priority:
        if name in candidates:
            return name
    for prefix in ('pass@', 'avg@'):
        matches = sorted(name for name in candidates if name.startswith(prefix))
        if matches:
            return matches[0]
    return sorted(candidates)[0]


async def _validate_columns(
    connection: asyncpg.Connection,
    required: Mapping[str, set[str]],
    *,
    label: str,
) -> None:
    rows = await connection.fetch(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = ANY($1::text[])
        """,
        list(required),
    )
    actual: dict[str, set[str]] = {table: set() for table in required}
    for row in rows:
        actual[str(row['table_name'])].add(str(row['column_name']))
    failures = []
    for table, columns in required.items():
        missing = sorted(columns - actual[table])
        if missing:
            failures.append(f"{table}: {', '.join(missing)}")
    if failures:
        raise MigrationError(f"{label} schema is incompatible: " + '; '.join(failures))


async def _validate_source_schema(connection: asyncpg.Connection) -> None:
    await _validate_columns(
        connection,
        LEGACY_REQUIRED_COLUMNS,
        label='legacy source',
    )


async def _validate_target_schema(connection: asyncpg.Connection) -> None:
    version = await connection.fetchval(
        """
        SELECT contract_version
        FROM evaluation_schema_metadata
        WHERE singleton = true
        """
    )
    if version != 4:
        raise MigrationError(f"target requires evaluation contract version 4, found {version!r}")


async def _database_identity(connection: asyncpg.Connection) -> tuple[Any, ...]:
    row = await connection.fetchrow(
        """
        SELECT current_database(), current_setting('port'),
               pg_postmaster_start_time()
        """
    )
    assert row is not None
    return tuple(row)


async def _fetch_score_rows(connection: asyncpg.Connection) -> list[dict[str, Any]]:
    rows = await connection.fetch(
        """
        SELECT s.score_id, s.task_id, s.cot_mode, s.metrics,
               s.created_at AS score_created_at,
               t.config_path, t.evaluator, t.is_param_search, t.is_tmp,
               t.created_at AS task_created_at, t.status AS task_status,
               t.git_hash, t.model_id, t.benchmark_id,
               t.desc AS task_description, t.sampling_config, t.log_path,
               m.data_version, m.arch_version, m.num_params, m.model_name,
               b.benchmark_name, b.benchmark_split,
               b.url AS benchmark_url, b.status AS benchmark_status,
               b.num_samples
        FROM scores AS s
        JOIN task AS t ON t.task_id = s.task_id
        JOIN model AS m ON m.model_id = t.model_id
        JOIN benchmark AS b ON b.benchmark_id = t.benchmark_id
        WHERE lower(t.status) = 'completed'
          AND NOT t.is_tmp
          AND NOT t.is_param_search
        ORDER BY s.created_at, s.score_id
        """
    )
    return [dict(row) for row in rows]


async def _fetch_details(
    connection: asyncpg.Connection,
    task_ids: Sequence[int],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not task_ids:
        return grouped
    rows = await connection.fetch(
        """
        SELECT c.completions_id, c.task_id, c.context, c.sample_index,
               c.avg_repeat_index, c.pass_index,
               c.created_at AS completion_created_at,
               c.status AS completion_status,
               e.eval_id, e.answer, e.ref_answer, e.is_passed,
               e.fail_reason, e.created_at AS eval_created_at,
               ch.checker_id, ch.answer_correct,
               ch.instruction_following_error, ch.world_knowledge_error,
               ch.math_error, ch.reasoning_logic_error,
               ch.thought_contains_correct_answer, ch.needs_human_review,
               ch.reason AS checker_reason,
               ch.created_at AS checker_created_at
        FROM completions AS c
        LEFT JOIN eval AS e ON e.completions_id = c.completions_id
        LEFT JOIN checker AS ch ON ch.completions_id = c.completions_id
        WHERE c.task_id = ANY($1::int[])
        ORDER BY c.task_id, c.sample_index, c.avg_repeat_index,
                 c.pass_index, c.completions_id
        """,
        list(task_ids),
    )
    for row in rows:
        item = dict(row)
        grouped[int(item['task_id'])].append(item)
    return grouped


async def _fetch_catalog_rows(connection: asyncpg.Connection, ) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rows = await connection.fetch(
        """
        SELECT catalog_id, benchmark_name, benchmark_split, field, source,
               source_family, target_kind, run_status, scope, metadata,
               created_at, updated_at
        FROM benchmark_catalog
        ORDER BY benchmark_name, benchmark_split, scope, catalog_id
        """
    )
    for row in rows:
        item = dict(row)
        key = (str(item['benchmark_name']), str(item['benchmark_split'] or ''))
        grouped[key].append(item)
    return grouped


def _benchmark_name(row: Mapping[str, Any]) -> str:
    name = str(row['benchmark_name'])
    split = str(row['benchmark_split'] or '')
    value = name if not split else f"{name}|{split}"
    if len(value) > 500:
        raise MigrationError(f"legacy benchmark name is too long for scoreboard-v1: {value!r}")
    return value


def _catalog_metadata_values(
    catalog_rows: Sequence[Mapping[str, Any]],
    *keys: str,
) -> Iterable[str]:
    for row in catalog_rows:
        metadata = row.get('metadata')
        if not isinstance(metadata, Mapping):
            continue
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                yield from (item for item in value if isinstance(item, str))


def _clean_strings(values: Iterable[Any]) -> list[str]:
    return sorted({value.strip() for value in values if isinstance(value, str) and value.strip()})


def _task_metadata(
    row: Mapping[str, Any],
    mapping: ModelMapping,
    catalog_rows: Sequence[Mapping[str, Any]],
) -> Task:
    task_name = _benchmark_name(row)
    tags = _clean_strings([
        'legacy-helicopter',
        row['evaluator'],
        *(catalog.get('field') for catalog in catalog_rows),
        *(catalog.get('source_family') for catalog in catalog_rows),
    ])
    languages = _clean_strings(_catalog_metadata_values(catalog_rows, 'language', 'languages'))
    split = str(row['benchmark_split'] or '')
    return Task.model_validate({
        'identity': (f"{mapping.weight_sha256}:{mapping.wkv_mode}:{task_name}"),
        'weight_sha256': mapping.weight_sha256,
        'weight_display_name': mapping.weight_display_name,
        'wkv_mode': mapping.wkv_mode,
        'benchmark': task_name,
        'task_name': task_name,
        'task_version': 'legacy',
        'dataset': str(row['benchmark_name']),
        'subset': split or None,
        'evaluation_splits': [split] if split else [],
        'languages': languages,
        'tags': tags,
    })


def _evaluation_record(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if row.get('eval_id') is None:
        return None
    return {
        'eval_id': int(row['eval_id']),
        'answer': _json_value(row.get('answer')),
        'reference_answer': _json_value(row.get('ref_answer')),
        'is_passed': bool(row['is_passed']),
        'fail_reason': _json_value(row.get('fail_reason')),
        'created_at': _json_value(row.get('eval_created_at')),
    }


def _checker_record(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if row.get('checker_id') is None:
        return None
    return {
        'checker_id': int(row['checker_id']),
        'answer_correct': bool(row['answer_correct']),
        'instruction_following_error': bool(row['instruction_following_error']),
        'world_knowledge_error': bool(row['world_knowledge_error']),
        'math_error': bool(row['math_error']),
        'reasoning_logic_error': bool(row['reasoning_logic_error']),
        'thought_contains_correct_answer': bool(row['thought_contains_correct_answer']),
        'needs_human_review': bool(row['needs_human_review']),
        'reason': _json_value(row.get('checker_reason')),
        'created_at': _json_value(row.get('checker_created_at')),
    }


def _samples(
    task_id: int,
    detail_rows: Sequence[Mapping[str, Any]],
) -> list[Sample]:
    by_sample: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for detail in detail_rows:
        by_sample[int(detail['sample_index'])].append(detail)
    samples: list[Sample] = []
    for new_index, legacy_index in enumerate(sorted(by_sample)):
        attempts: list[dict[str, Any]] = []
        pass_values: list[bool] = []
        for detail in by_sample[legacy_index]:
            evaluation = _evaluation_record(detail)
            checker = _checker_record(detail)
            if evaluation is not None:
                pass_values.append(bool(evaluation['is_passed']))
            attempt = {
                'completion_id': int(detail['completions_id']),
                'avg_repeat_index': int(detail['avg_repeat_index']),
                'pass_index': int(detail['pass_index']),
                'status': str(detail['completion_status']),
                'created_at': _json_value(detail['completion_created_at']),
                'context': _json_value(detail['context']),
                'evaluation': evaluation,
                'checker': checker,
            }
            attempts.append(attempt)
        metrics: dict[str, Any] = {
            'legacy_attempt_count': len(attempts),
            'legacy_evaluated_attempt_count': len(pass_values),
        }
        if pass_values:
            metrics['legacy_pass_rate'] = sum(pass_values) / len(pass_values)
        samples.append(
            Sample.model_validate({
                'sample_index': new_index,
                'document_index': max(0, legacy_index),
                'document': {
                    'legacy_task_id': task_id,
                    'legacy_sample_index': legacy_index,
                },
                'metrics': metrics,
                'model_response': {
                    'attempts': attempts
                },
            })
        )
    return samples


def _source_config_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return json_sha256([{
        'score_id': int(row['score_id']),
        'task_id': int(row['task_id']),
        'config_path': row['config_path'],
        'git_hash': row['git_hash'],
        'cot_mode': row['cot_mode'],
        'sampling_config': _json_value(row['sampling_config']),
    } for row in sorted(rows, key=lambda item: int(item['score_id']))])


def _registry_hash(
    rows: Sequence[Mapping[str, Any]],
    catalogs: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> str:
    relevant: dict[int, Any] = {}
    for row in rows:
        key = (str(row['benchmark_name']), str(row['benchmark_split'] or ''))
        for catalog in catalogs.get(key, []):
            relevant[int(catalog['catalog_id'])] = _json_value(catalog)
    return json_sha256([relevant[key] for key in sorted(relevant)])


def _build_publication(
    *,
    campaign_id: uuid.UUID,
    row: Mapping[str, Any],
    task: Task,
    mapping: ModelMapping,
    catalog_rows: Sequence[Mapping[str, Any]],
    details: Sequence[Mapping[str, Any]],
    campaign_kind: str,
) -> TaskPublication:
    metrics, raw_metrics = split_scalar_metrics(row['metrics'])
    primary_metric = choose_primary_metric(metrics)
    sampling_config = _json_value(row['sampling_config'])
    if not isinstance(sampling_config, dict):
        sampling_config = {'legacy_value': sampling_config}
    diagnostics = {
        'migration': {
            'name': MIGRATION_NAME,
            'version': MIGRATION_VERSION,
            'campaign_kind': campaign_kind,
        },
        'legacy': {
            'score_id': int(row['score_id']),
            'task_id': int(row['task_id']),
            'model_id': int(row['model_id']),
            'benchmark_id': int(row['benchmark_id']),
            'score_created_at': _json_value(row['score_created_at']),
            'task_created_at': _json_value(row['task_created_at']),
            'task_status': str(row['task_status']),
            'config_path': _json_value(row['config_path']),
            'log_path': _json_value(row['log_path']),
            'git_hash': str(row['git_hash']),
            'benchmark_url': _json_value(row['benchmark_url']),
            'benchmark_status': str(row['benchmark_status']),
            'benchmark_num_samples': int(row['num_samples']),
            'catalog': _json_value(catalog_rows),
            **raw_metrics,
        },
        'metadata_sources': {
            'weight_sha256': 'explicit-model-map',
            'wkv_mode': mapping.wkv_mode_source,
        },
    }
    publication = {
        'schema_version': 'scoreboard-v1',
        'campaign_id': str(campaign_id),
        'task': task.model_dump(mode='json'),
        'result_files': [],
        'task_config': {
            'legacy_evaluator': str(row['evaluator']),
            'legacy_cot_mode': str(row['cot_mode']),
            'legacy_config_path': _json_value(row['config_path']),
            'legacy_git_hash': str(row['git_hash']),
            'legacy_description': _json_value(row['task_description']),
        },
        'environment': {
            'framework': EVALUATOR_SOURCES[str(row['evaluator'])],
            'legacy_evaluator': str(row['evaluator']),
            'legacy_model': {
                'model_name': str(row['model_name']),
                'arch_version': str(row['arch_version']),
                'data_version': str(row['data_version']),
                'num_params': str(row['num_params']),
            },
        },
        'sampling_config': sampling_config,
        'primary_metric': primary_metric,
        'metrics': metrics,
        'diagnostics': diagnostics,
        'samples': [sample.model_dump(mode='json') for sample in _samples(int(row['task_id']), details)],
    }
    return TaskPublication.model_validate(publication)


def _latest_score_ids(rows: Sequence[Mapping[str, Any]]) -> set[int]:
    latest: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        source = EVALUATOR_SOURCES[str(row['evaluator'])]
        key = (source, str(row['model_name']), _benchmark_name(row))
        existing = latest.get(key)
        if existing is None or (row['score_created_at'],
                                int(row['score_id'])) > (existing['score_created_at'], int(existing['score_id'])):
            latest[key] = row
    return {int(row['score_id']) for row in latest.values()}


def _campaign_plan(
    *,
    rows: Sequence[Mapping[str, Any]],
    source: str,
    kind: str,
    mappings: Mapping[str, ModelMapping],
    catalogs: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    details: Mapping[int, Sequence[Mapping[str, Any]]],
) -> CampaignPlan:
    task_metadata: list[Task] = []
    catalog_by_score: dict[int, Sequence[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row['benchmark_name']), str(row['benchmark_split'] or ''))
        catalog_rows = catalogs.get(key, [])
        catalog_by_score[int(row['score_id'])] = catalog_rows
        task_metadata.append(_task_metadata(row, mappings[str(row['model_name'])], catalog_rows))
    identities = [task.identity for task in task_metadata]
    if len(identities) != len(set(identities)):
        duplicates = sorted(identity for identity in set(identities) if identities.count(identity) > 1)
        raise MigrationError('campaign would contain duplicate task identities: ' + ', '.join(duplicates))
    benchmarks = sorted({task.benchmark for task in task_metadata})
    rerun_reason = None
    if kind == 'history':
        rerun_reason = f"Imported legacy Helicopter score_id={rows[0]['score_id']}"
    campaign_data = {
        'schema_version': 'scoreboard-v1',
        'source': source,
        'config_sha256': _source_config_hash(rows),
        'registry_sha256': _registry_hash(rows, catalogs),
        'contract_sha256': CONTRACT_SHA256,
        'configured_benchmarks': benchmarks,
        'resolved_benchmarks': benchmarks,
        'skipped_benchmarks': [],
        'expected_tasks': [task.model_dump(mode='json') for task in task_metadata],
        'rerun_reason': rerun_reason,
    }
    campaign_data['run_key'] = campaign_run_key(campaign_data)
    campaign = CampaignCreate.model_validate(campaign_data)
    campaign_id = uuid.uuid5(NAMESPACE, f"campaign:{campaign.run_key}")
    task_plans: list[TaskPlan] = []
    for row, task in zip(rows, task_metadata, strict=True):
        publication = _build_publication(
            campaign_id=campaign_id,
            row=row,
            task=task,
            mapping=mappings[str(row['model_name'])],
            catalog_rows=catalog_by_score[int(row['score_id'])],
            details=details.get(int(row['task_id']), []),
            campaign_kind=kind,
        )
        content_sha256 = json_sha256(publication.model_dump(mode='json'))
        evaluation_id = uuid.uuid5(
            NAMESPACE,
            f"evaluation:{campaign_id}:{task.identity}",
        )
        task_plans.append(
            TaskPlan(
                evaluation_id=evaluation_id,
                source_score_id=int(row['score_id']),
                created_at=row['task_created_at'],
                content_sha256=content_sha256,
                publication=publication,
            )
        )
    return CampaignPlan(
        campaign_id=campaign_id,
        campaign=campaign,
        created_at=min(row['task_created_at'] for row in rows),
        completed_at=max(row['score_created_at'] for row in rows),
        tasks=tuple(task_plans),
        kind=kind,
    )


async def build_migration_plan(
    source: asyncpg.Connection,
    mappings: Mapping[str, ModelMapping],
) -> MigrationPlan:
    """Read a consistent legacy snapshot and construct deterministic target rows."""

    await _validate_source_schema(source)
    source_task_count = int(await source.fetchval('SELECT count(*) FROM task'))
    rows = await _fetch_score_rows(source)
    if not rows:
        raise MigrationError('legacy source has no eligible completed scores')
    unsupported = sorted({str(row['evaluator']) for row in rows if str(row['evaluator']) not in EVALUATOR_SOURCES})
    if unsupported:
        raise MigrationError('legacy source contains unsupported evaluators: ' + ', '.join(unsupported))
    selected_models = {str(row['model_name']) for row in rows}
    missing_models = sorted(selected_models - set(mappings))
    if missing_models:
        raise MigrationError('model mapping is missing selected models: ' + ', '.join(missing_models))
    catalogs = await _fetch_catalog_rows(source)
    details = await _fetch_details(source, [int(row['task_id']) for row in rows])
    latest_ids = _latest_score_ids(rows)
    campaigns: list[CampaignPlan] = []
    for source_name in sorted(set(EVALUATOR_SOURCES.values())):
        snapshot_rows = [
            row for row in rows
            if EVALUATOR_SOURCES[str(row['evaluator'])] == source_name and int(row['score_id']) in latest_ids
        ]
        if snapshot_rows:
            campaigns.append(
                _campaign_plan(
                    rows=snapshot_rows,
                    source=source_name,
                    kind='snapshot',
                    mappings=mappings,
                    catalogs=catalogs,
                    details=details,
                )
            )
    for row in rows:
        if int(row['score_id']) in latest_ids:
            continue
        campaigns.append(
            _campaign_plan(
                rows=[row],
                source=EVALUATOR_SOURCES[str(row['evaluator'])],
                kind='history',
                mappings=mappings,
                catalogs=catalogs,
                details=details,
            )
        )
    campaigns.sort(
        key=lambda campaign: (
            campaign.completed_at,
            campaign.kind != 'history',
            campaign.campaign.run_key,
        )
    )
    return MigrationPlan(
        campaigns=tuple(campaigns),
        source_task_count=source_task_count,
        selected_score_count=len(rows),
        excluded_task_count=source_task_count - len(rows),
    )


def _mismatches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    return [name for name, value in expected.items() if actual.get(name) != value]


def _campaign_expected(plan: CampaignPlan) -> dict[str, Any]:
    campaign = plan.campaign
    return {
        'id': plan.campaign_id,
        'run_key': campaign.run_key,
        'status': 'complete',
        'source': campaign.source,
        'config_sha256': campaign.config_sha256,
        'registry_sha256': campaign.registry_sha256,
        'contract_sha256': campaign.contract_sha256,
        'configured_benchmarks': campaign.configured_benchmarks,
        'resolved_benchmarks': campaign.resolved_benchmarks,
        'skipped_benchmarks': campaign.skipped_benchmarks,
        'expected_tasks': [task.model_dump(mode='json') for task in campaign.expected_tasks],
        'rerun_reason': campaign.rerun_reason,
        'publisher': PUBLISHER,
        'created_at': plan.created_at,
        'completed_at': plan.completed_at,
    }


def _task_expected(campaign_id: uuid.UUID, plan: TaskPlan) -> dict[str, Any]:
    publication = plan.publication
    return {
        'id': plan.evaluation_id,
        'campaign_id': campaign_id,
        'task_identity': publication.task.identity,
        'content_sha256': plan.content_sha256,
        'task': publication.task.model_dump(mode='json'),
        'result_files': [item.model_dump(mode='json') for item in publication.result_files],
        'task_config': publication.task_config,
        'environment': publication.environment,
        'sampling_config': publication.sampling_config,
        'primary_metric': publication.primary_metric,
        'metrics': publication.metrics,
        'diagnostics': publication.diagnostics,
        'created_at': plan.created_at,
    }


async def _assert_campaign_matches(
    target: asyncpg.Connection,
    plan: CampaignPlan,
) -> None:
    rows = await target.fetch(
        """
        SELECT id, run_key, status, source, config_sha256, registry_sha256,
               contract_sha256, configured_benchmarks, resolved_benchmarks,
               skipped_benchmarks, expected_tasks, rerun_reason, publisher,
               created_at, completed_at
        FROM evaluation_campaign
        WHERE run_key = $1 OR id = $2
        """,
        plan.campaign.run_key,
        plan.campaign_id,
    )
    if not rows:
        raise MigrationConflictError(f"target campaign is missing: {plan.campaign.run_key}")
    if len(rows) != 1:
        raise MigrationConflictError(f"target campaign id/run-key collision: {plan.campaign.run_key}")
    row = rows[0]
    campaign_mismatches = _mismatches(dict(row), _campaign_expected(plan))
    if campaign_mismatches:
        raise MigrationConflictError(
            f"target campaign {plan.campaign.run_key} differs in: " + ', '.join(campaign_mismatches)
        )
    for task_plan in plan.tasks:
        publication = task_plan.publication
        task_rows = await target.fetch(
            """
            SELECT id, campaign_id, task_identity, content_sha256, task,
                   result_files, task_config, environment, sampling_config,
                   primary_metric, metrics, diagnostics, created_at
            FROM evaluation_task
            WHERE (campaign_id = $1 AND task_identity = $2) OR id = $3
            """,
            plan.campaign_id,
            publication.task.identity,
            task_plan.evaluation_id,
        )
        if not task_rows:
            raise MigrationConflictError(f"target task is missing: {publication.task.identity}")
        if len(task_rows) != 1:
            raise MigrationConflictError(f"target task id/identity collision: {publication.task.identity}")
        task_row = task_rows[0]
        task_mismatches = _mismatches(dict(task_row), _task_expected(plan.campaign_id, task_plan))
        if task_mismatches:
            raise MigrationConflictError(
                f"target task {publication.task.identity} differs in: " + ', '.join(task_mismatches)
            )
        sample_rows = await target.fetch(
            """
            SELECT sample_index, document_index, document, metrics, model_response
            FROM evaluation_sample
            WHERE evaluation_id = $1
            ORDER BY sample_index
            """,
            task_plan.evaluation_id,
        )
        expected_samples = [sample.model_dump(mode='json') for sample in publication.samples]
        if [dict(sample) for sample in sample_rows] != expected_samples:
            raise MigrationConflictError(f"target samples differ for task {publication.task.identity}")


async def _classify_campaigns(
    target: asyncpg.Connection,
    plan: MigrationPlan,
) -> tuple[list[CampaignPlan], list[CampaignPlan]]:
    create: list[CampaignPlan] = []
    reuse: list[CampaignPlan] = []
    for campaign in plan.campaigns:
        rows = await target.fetch(
            """
            SELECT id
            FROM evaluation_campaign
            WHERE run_key = $1 OR id = $2
            """,
            campaign.campaign.run_key,
            campaign.campaign_id,
        )
        if not rows:
            colliding_task = await target.fetchval(
                """
                SELECT id
                FROM evaluation_task
                WHERE id = ANY($1::uuid[])
                LIMIT 1
                """,
                [task.evaluation_id for task in campaign.tasks],
            )
            if colliding_task is not None:
                raise MigrationConflictError(f"target evaluation id collision: {colliding_task}")
            create.append(campaign)
            continue
        if len(rows) != 1:
            raise MigrationConflictError('target campaign id/run-key collision: '
                                         f"{campaign.campaign.run_key}")
        await _assert_campaign_matches(target, campaign)
        reuse.append(campaign)
    return create, reuse


async def _insert_campaign(
    target: asyncpg.Connection,
    plan: CampaignPlan,
) -> None:
    expected = _campaign_expected(plan)
    await target.execute(
        """
        INSERT INTO evaluation_campaign (
            id, run_key, status, source, config_sha256, registry_sha256,
            contract_sha256, configured_benchmarks, resolved_benchmarks,
            skipped_benchmarks, expected_tasks, rerun_reason, publisher,
            created_at, completed_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
            $14, $15
        )
        """,
        *expected.values(),
    )
    for task_plan in plan.tasks:
        task_expected = _task_expected(plan.campaign_id, task_plan)
        await target.execute(
            """
            INSERT INTO evaluation_task (
                id, campaign_id, task_identity, content_sha256, task,
                result_files, task_config, environment, sampling_config,
                primary_metric, metrics, diagnostics, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
            )
            """,
            *task_expected.values(),
        )
        samples = task_plan.publication.samples
        for start in range(0, len(samples), 500):
            await target.executemany(
                """
                INSERT INTO evaluation_sample (
                    evaluation_id, sample_index, document_index,
                    document, metrics, model_response
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                [(
                    task_plan.evaluation_id,
                    sample.sample_index,
                    sample.document_index,
                    sample.document,
                    sample.metrics,
                    sample.model_response,
                ) for sample in samples[start:start + 500]],
            )


def _summary(
    plan: MigrationPlan,
    create: Sequence[CampaignPlan],
    reuse: Sequence[CampaignPlan],
    *,
    apply: bool,
) -> MigrationSummary:
    return MigrationSummary(
        mode='apply' if apply else 'dry-run',
        source_task_count=plan.source_task_count,
        selected_score_count=plan.selected_score_count,
        excluded_task_count=plan.excluded_task_count,
        planned_campaign_count=len(plan.campaigns),
        planned_task_count=plan.task_count,
        planned_sample_count=plan.sample_count,
        planned_attempt_count=plan.attempt_count,
        snapshot_campaign_count=sum(campaign.kind == 'snapshot' for campaign in plan.campaigns),
        historical_campaign_count=sum(campaign.kind == 'history' for campaign in plan.campaigns),
        campaigns_to_create=len(create),
        campaigns_reused=len(reuse),
        tasks_to_create=sum(len(campaign.tasks) for campaign in create),
        tasks_reused=sum(len(campaign.tasks) for campaign in reuse),
    )


async def migrate_legacy_scoreboard(
    source: asyncpg.Connection,
    target: asyncpg.Connection,
    mappings: Mapping[str, ModelMapping],
    *,
    apply: bool = False,
) -> MigrationSummary:
    """Plan, validate, and optionally apply one atomic legacy migration."""

    await _configure_connection(source)
    await _configure_connection(target)
    if await _database_identity(source) == await _database_identity(target):
        raise MigrationError('legacy source and target must be different databases')
    async with source.transaction(isolation='repeatable_read', readonly=True):
        plan = await build_migration_plan(source, mappings)
    await _validate_target_schema(target)
    if not apply:
        async with target.transaction(isolation='repeatable_read', readonly=True):
            create, reuse = await _classify_campaigns(target, plan)
        return _summary(plan, create, reuse, apply=False)

    async with target.transaction(isolation='serializable'):
        await target.execute(
            'SELECT pg_advisory_xact_lock(hashtextextended($1, 0))',
            MIGRATION_NAME,
        )
        create, reuse = await _classify_campaigns(target, plan)
        for campaign in create:
            await _insert_campaign(target, campaign)
        for campaign in plan.campaigns:
            await _assert_campaign_matches(target, campaign)
    async with target.transaction(isolation='repeatable_read', readonly=True):
        for campaign in plan.campaigns:
            await _assert_campaign_matches(target, campaign)
    return _summary(plan, create, reuse, apply=True)


def _env_database_kwargs(prefix: str) -> dict[str, Any]:
    dsn = os.environ.get(f"{prefix}_SCOREBOARD_DSN") or os.environ.get(f"{prefix}_DATABASE_URL")
    if dsn:
        return {'dsn': dsn, 'timeout': 30}
    names = {
        'host': f"{prefix}_SCOREBOARD_DB_HOST",
        'port': f"{prefix}_SCOREBOARD_DB_PORT",
        'user': f"{prefix}_SCOREBOARD_DB_USER",
        'password': f"{prefix}_SCOREBOARD_DB_PASSWORD",
        'database': f"{prefix}_SCOREBOARD_DB_NAME",
    }
    values = {key: os.environ.get(name) for key, name in names.items()}
    missing = [names[key] for key in ('host', 'port', 'user', 'database') if not values[key]]
    if missing:
        raise MigrationError('missing database settings: ' + ', '.join(missing))
    try:
        values['port'] = int(str(values['port']))
    except ValueError as error:
        raise MigrationError(f"{names['port']} must be an integer") from error
    values['timeout'] = 30
    return values


async def _run(args: argparse.Namespace) -> MigrationSummary:
    mappings = load_model_mappings(args.model_map)
    source = await asyncpg.connect(**_env_database_kwargs('LEGACY'))
    try:
        target = await asyncpg.connect(**_env_database_kwargs('TARGET'))
        try:
            return await migrate_legacy_scoreboard(
                source,
                target,
                mappings,
                apply=args.apply,
            )
        finally:
            await target.close()
    finally:
        await source.close()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Migrate legacy Helicopter scoreboard data into contract v4', )
    parser.add_argument(
        '--model-map',
        required=True,
        type=Path,
        help='JSON mapping from legacy model names to real weight metadata',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='write atomically after validation (default: read-only dry-run)',
    )
    return parser.parse_args()


def main() -> None:
    """Run the migration CLI and print its machine-readable summary."""

    try:
        summary = asyncio.run(_run(_arguments()))
    except (MigrationError, ValueError, asyncpg.PostgresError) as error:
        raise SystemExit(f"migration failed: {error}") from error
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
