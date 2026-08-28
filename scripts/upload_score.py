"""Migrate legacy EvalScope scoreboard rows to a new Scoreboard over HTTP."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
from itertools import chain, groupby
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen

SCHEMA_VERSION = 'scoreboard-v1'
MIGRATION_CONTRACT = {
    'name': 'legacy-evalscope-scoreboard',
    'source': 'postgresql-read-only',
    'target': 'scoreboard-v1-http',
    'version': 1,
}

SOURCE_QUERY = r'''
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
COPY (
    SELECT payload
    FROM (
        SELECT 0 AS kind_order, model_id AS task_order, 0 AS sample_order,
               jsonb_build_object(
                   'kind', 'model',
                   'model_id', model_id,
                   'model_name', model_name,
                   'arch_version', arch_version,
                   'data_version', data_version,
                   'num_params', num_params
               ) AS payload
        FROM model
        UNION ALL
        SELECT 1, t.task_id, 0,
               jsonb_build_object(
                   'kind', 'task',
                   'task_id', t.task_id,
                   'score_id', s.score_id,
                   'model_id', m.model_id,
                   'model_name', m.model_name,
                   'arch_version', m.arch_version,
                   'data_version', m.data_version,
                   'num_params', m.num_params,
                   'benchmark_id', b.benchmark_id,
                   'benchmark_name', b.benchmark_name,
                   'benchmark_split', b.benchmark_split,
                   'benchmark_samples', b.num_samples,
                   'evaluator', t.evaluator,
                   'config_path', t.config_path,
                   'git_hash', t.git_hash,
                   'description', t.desc,
                   'sampling_config', t.sampling_config,
                   'log_path', t.log_path,
                   'task_created_at', t.created_at,
                   'cot_mode', s.cot_mode,
                   'metrics', s.metrics,
                   'score_created_at', s.created_at
               )
        FROM task t
        JOIN scores s USING (task_id)
        JOIN model m USING (model_id)
        JOIN benchmark b USING (benchmark_id)
        WHERE lower(t.evaluator) LIKE 'evalscope%'
          AND lower(t.status) = 'completed'
          AND NOT t.is_tmp
          AND NOT t.is_param_search
        UNION ALL
        SELECT 2, t.task_id, c.sample_index,
               jsonb_build_object(
                   'kind', 'sample',
                   'task_id', t.task_id,
                   'completion_id', c.completions_id,
                   'sample_index', c.sample_index,
                   'repeat_index', c.avg_repeat_index,
                   'pass_index', c.pass_index,
                   'completion_status', c.status,
                   'messages', (
                       SELECT coalesce(jsonb_agg(entry.value ORDER BY entry.ordinality), '[]'::jsonb)
                       FROM jsonb_array_elements(
                           coalesce(c.context #> '{agent_result,review,messages}', '[]'::jsonb)
                       ) WITH ORDINALITY AS entry(value, ordinality)
                       WHERE entry.value ->> 'role' IS DISTINCT FROM 'assistant'
                   ),
                   'sample_score', jsonb_build_object(
                       'group_id', c.context #> '{agent_result,review,sample_score,group_id}',
                       'score', jsonb_build_object(
                           'prediction', c.context #> '{agent_result,review,sample_score,score,prediction}',
                           'extracted_prediction',
                               c.context #> '{agent_result,review,sample_score,score,extracted_prediction}',
                           'target', c.context #> '{agent_result,review,sample_score,score,target}',
                           'value', c.context #> '{agent_result,review,sample_score,score,value}'
                       )
                   ),
                   'model_output', jsonb_build_object(
                       'choices', jsonb_build_array(jsonb_build_object(
                           'message', c.context #> '{agent_result,prediction,model_output,choices,0,message}',
                           'stop_reason',
                               c.context #> '{agent_result,prediction,model_output,choices,0,stop_reason}'
                       )),
                       'usage', c.context #> '{agent_result,prediction,model_output,usage}'
                   ),
                   'context_audit', jsonb_build_object(
                       'finish_reason', c.context #> '{agent_result,context_audit,finish_reason}'
                   ),
                   'source_index', c.context #> '{agent_result,source_index}',
                   'subset', c.context #> '{agent_result,subset}',
                   'eval_id', e.eval_id,
                   'answer', e.answer,
                   'reference_answer', e.ref_answer,
                   'is_passed', e.is_passed,
                   'fail_reason', e.fail_reason
               )
        FROM task t
        JOIN scores s USING (task_id)
        JOIN completions c USING (task_id)
        LEFT JOIN eval e USING (completions_id)
        WHERE lower(t.evaluator) LIKE 'evalscope%'
          AND lower(t.status) = 'completed'
          AND NOT t.is_tmp
          AND NOT t.is_param_search
    ) rows
    ORDER BY kind_order, task_order, sample_order,
             (payload ->> 'repeat_index')::integer NULLS FIRST,
             (payload ->> 'pass_index')::integer NULLS FIRST,
             (payload ->> 'completion_id')::integer NULLS FIRST
) TO STDOUT WITH (FORMAT CSV);
COMMIT;
'''


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def campaign_run_key(campaign: Mapping[str, Any]) -> str:
    payload = dict(campaign)
    payload.pop('run_key', None)
    for name in ('configured_benchmarks', 'resolved_benchmarks', 'skipped_benchmarks'):
        payload[name] = sorted(payload[name])
    tasks = []
    for value in payload['expected_tasks']:
        task = dict(value)
        for name in ('evaluation_splits', 'languages', 'tags'):
            task[name] = sorted(task[name])
        tasks.append(task)
    payload['expected_tasks'] = sorted(tasks, key=lambda task: (task['identity'], canonical_json(task)))
    return json_sha256(payload)


def send(
    base_url: str,
    token: str,
    method: str,
    path: str,
    idempotency_key: str,
    payload: Mapping[str, Any] | None = None,
    *,
    opener: Callable[..., Any] = urlopen,
) -> tuple[int, dict[str, Any]]:
    body = canonical_json(payload) if payload is not None else None
    headers = {'Authorization': f'Bearer {token}', 'Idempotency-Key': idempotency_key}
    if body is not None:
        headers.update({'Content-Type': 'application/json', 'Content-Encoding': 'gzip'})
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=gzip.compress(body) if body is not None else None,
        method=method,
        headers=headers,
    )
    with opener(request, timeout=60) as response:
        return response.status, json.loads(response.read().decode('utf-8'))


def publish_bundle(
    campaign: Mapping[str, Any],
    tasks: Iterable[dict[str, Any]],
    base_url: str,
    token: str,
    *,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    campaign_status, campaign_receipt = send(
        base_url,
        token,
        'POST',
        '/api/v1/evaluation-campaigns',
        f"campaign:{campaign['run_key']}",
        campaign,
        opener=opener,
    )
    campaign_id = campaign_receipt['campaign_id']
    if campaign_receipt['status'] == 'complete':
        return {
            'campaign': {
                'http_status': campaign_status,
                **campaign_receipt
            },
            'tasks': [],
            'finalize': None,
        }
    task_receipts = []
    for task in tasks:
        publication = {**task, 'campaign_id': campaign_id}
        body = canonical_json(publication)
        identity = quote(publication['task']['identity'], safe='')
        task_status, receipt = send(
            base_url,
            token,
            'PUT',
            f'/api/v1/evaluation-campaigns/{campaign_id}/tasks/{identity}',
            f'publish:{hashlib.sha256(body).hexdigest()}',
            publication,
            opener=opener,
        )
        task_receipts.append({'http_status': task_status, **receipt})
        task = publication = body = None
    finalize_status, finalize_receipt = send(
        base_url,
        token,
        'POST',
        f'/api/v1/evaluation-campaigns/{campaign_id}/finalize',
        f'finalize:{campaign_id}',
        opener=opener,
    )
    return {
        'campaign': {
            'http_status': campaign_status,
            **campaign_receipt
        },
        'tasks': task_receipts,
        'finalize': {
            'http_status': finalize_status,
            **finalize_receipt
        },
    }


def _psql_command() -> tuple[list[str], dict[str, str]]:
    command = ['psql', '-X', '-qAt', '--set', 'ON_ERROR_STOP=1']
    dsn = os.environ.get('LEGACY_SCOREBOARD_DSN')
    if dsn:
        command.extend(['--dbname', dsn])
    else:
        names = {
            'host': 'LEGACY_SCOREBOARD_DB_HOST',
            'port': 'LEGACY_SCOREBOARD_DB_PORT',
            'user': 'LEGACY_SCOREBOARD_DB_USER',
            'dbname': 'LEGACY_SCOREBOARD_DB_NAME',
        }
        missing = [name for name in names.values() if not os.environ.get(name)]
        if missing:
            raise ValueError(f"missing legacy database settings: {', '.join(missing)}")
        for option, name in names.items():
            command.extend([f'--{option}', os.environ[name]])
    environment = dict(os.environ)
    password = os.environ.get('LEGACY_SCOREBOARD_DB_PASSWORD')
    if password:
        environment['PGPASSWORD'] = password
    return command, environment


def read_source_rows(
    *,
    popen: Callable[..., Any] = subprocess.Popen,
) -> Iterable[dict[str, Any]]:
    command, environment = _psql_command()
    process = popen(
        [*command, '--command', SOURCE_QUERY],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        env=environment,
    )
    assert process.stdout is not None
    csv.field_size_limit(2**31 - 1)
    try:
        for record in csv.reader(process.stdout):
            if record:
                yield json.loads(record[0])
    except BaseException:
        process.terminate()
        process.communicate()
        raise
    _, error = process.communicate()
    if process.returncode:
        raise RuntimeError(f'legacy database query failed: {error.strip()}')


def load_model_mappings(path: Path) -> dict[str, dict[str, str]]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or not value:
        raise ValueError('model map must be a non-empty JSON object')
    mappings = {}
    for model_name, mapping in value.items():
        if not isinstance(mapping, dict):
            raise ValueError(f'model map entry {model_name!r} must be an object')
        sha256 = mapping.get('weight_sha256')
        mode = mapping.get('wkv_mode')
        display_name = mapping.get('weight_display_name')
        if not isinstance(sha256, str) or re.fullmatch(r'[0-9a-f]{64}', sha256) is None:
            raise ValueError(f'model {model_name!r} requires a lowercase SHA-256 weight digest')
        if mode not in {'fp16', 'fp32io16'}:
            raise ValueError(f'model {model_name!r} requires wkv_mode fp16 or fp32io16')
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError(f'model {model_name!r} requires weight_display_name')
        mappings[model_name] = {
            'weight_sha256': sha256,
            'weight_display_name': display_name,
            'wkv_mode': mode,
        }
    return mappings


def _normalized_parameters(value: Any) -> str:
    return str(value).replace('_', '.').upper()


def _model_variant(model: Mapping[str, Any]) -> dict[str, str]:
    return {
        'label': str(model['model_name']),
        'architecture': str(model['arch_version']).upper(),
        'generation': str(model['data_version']).upper(),
        'parameters': _normalized_parameters(model['num_params']),
    }


def _comparison_coordinate(
    task: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]] | None:
    parameters = _normalized_parameters(task['num_params'])
    candidates = {(str(model['data_version']), str(model['model_name'])): model
                  for model in models
                  if str(model['arch_version']) == str(task['arch_version'])
                  and _normalized_parameters(model['num_params']) == parameters}
    generations = sorted({generation for generation, _ in candidates})
    if len(generations) < 2 or str(task['data_version']) not in generations[-2:]:
        return None
    a_generation, b_generation = generations[-2:]
    a_model = next(model for (generation, _), model in candidates.items() if generation == a_generation)
    b_model = next(model for (generation, _), model in candidates.items() if generation == b_generation)
    selected_arm = 'a' if str(task['data_version']) == a_generation else 'b'
    selected_model = a_model if selected_arm == 'a' else b_model
    group_id = re.sub(r'[^a-z0-9_.-]+', '_', parameters.lower())
    coordinate = {
        'comparison': {
            'id': 'generation',
            'label': f'{a_generation.upper()} vs {b_generation.upper()}',
            'short_label': 'Generation',
            'a_label': a_generation.upper(),
            'b_label': b_generation.upper(),
            'contract': 'The legacy scoreboard pairs the latest two model data versions at the same parameter size.',
        },
        'parameter_group': {
            'id': group_id,
            'label': parameters,
            'a_model': _model_variant(a_model),
            'b_model': _model_variant(b_model),
            'parameter_delta_percent': 0.0,
            'comparable': True,
        },
        'arm': selected_arm,
    }
    return coordinate, _model_variant(selected_model)


def _text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return canonical_json(value).decode('utf-8')


def _output_fields(row: Mapping[str, Any]) -> tuple[str, str, int, float | None, dict[str, Any] | None]:
    output = row.get('model_output') or {}
    choices = output.get('choices') or []
    choice = choices[0] if choices else {}
    message = choice.get('message') or {}
    score = (row.get('sample_score') or {}).get('score') or {}
    raw_completion = _text(score.get('prediction') or row.get('answer') or message.get('content'))
    stop_reason = str(choice.get('stop_reason') or (row.get('context_audit') or {}).get('finish_reason') or 'unknown')
    usage = output.get('usage') or {}
    perf = message.get('perf_metrics') or {}
    generated_tokens = int(usage.get('output_tokens') or perf.get('output_tokens') or 0)
    latency = perf.get('latency')
    latency_ms = float(latency) * 1000 if isinstance(latency, (int, float)) and math.isfinite(latency) else None
    return raw_completion, stop_reason, generated_tokens, latency_ms, message or None


def _sample(
    sample_index: int,
    row: Mapping[str, Any],
    include_answer: bool,
) -> tuple[dict[str, Any], bool]:
    messages = row.get('messages') or []
    prompt_messages = [message for message in messages if message.get('role') != 'assistant']
    sample_score = row.get('sample_score') or {}
    score = sample_score.get('score') or {}
    score_values = score.get('value') or {}
    raw_completion, stop_reason, generated_tokens, latency_ms, output_message = _output_fields(row)
    has_eval = row.get('eval_id') is not None
    if not has_eval:
        outcome = 'unanswered'
    elif row.get('is_passed'):
        outcome = 'correct'
    elif not _text(row.get('answer')).strip():
        outcome = 'unanswered'
    else:
        outcome = 'incorrect'
    answer = None
    if include_answer:
        group_id = sample_score.get('group_id')
        answer = {
            'outcome': outcome,
            'problem_id': str(group_id if group_id is not None else row['sample_index']),
            'repeat_id': int(row['repeat_index']),
            'ground_truth': _text(row.get('reference_answer') or score.get('target')),
            'extracted_answer': _text(score.get('extracted_prediction') or row.get('answer')),
            'assembled_prompt': canonical_json(prompt_messages).decode('utf-8'),
            'raw_completion': raw_completion,
            'fail_reason': _text(row.get('fail_reason')).strip() or None,
            'generated_tokens': generated_tokens,
            'latency_ms': latency_ms,
        }
    sample = {
        'sample_index': sample_index,
        'document_index': int(row['sample_index']),
        'document': {
            'source_sample_index': int(row['sample_index']),
            'source_index': row.get('source_index'),
            'repeat_index': int(row['repeat_index']),
            'pass_index': int(row['pass_index']),
            'subset': row.get('subset'),
        },
        'metrics': {
            **score_values, 'has_eval_record': has_eval,
            'is_passed': row.get('is_passed')
        },
        'model_response': {
            'text': raw_completion,
            'stop_reason': stop_reason,
            'message': output_message,
        },
        'answer': answer,
    }
    return sample, stop_reason in {'max_tokens', 'model_length'}


def _scalar_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): float(metric)
        for name, metric in value.items()
        if not isinstance(metric, bool) and isinstance(metric, (int, float)) and math.isfinite(float(metric))
    }


def _task_identity(task: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    benchmark = str(task['benchmark_name'])
    split = str(task.get('benchmark_split') or '')
    task_name = f"{benchmark}#{task['task_id']}"
    return {
        'identity': f"{mapping['weight_sha256']}:{mapping['wkv_mode']}:{task_name}",
        'weight_sha256': mapping['weight_sha256'],
        'weight_display_name': mapping['weight_display_name'],
        'wkv_mode': mapping['wkv_mode'],
        'benchmark': benchmark,
        'task_name': task_name,
        'task_version': str(task['task_id']),
        'dataset': benchmark,
        'subset': split or None,
        'evaluation_splits': [split] if split else [],
        'languages': [],
        'tags': [],
    }


def _task_publication(
    task: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    benchmark = str(task['benchmark_name'])
    coordinate = _comparison_coordinate(task, models)
    samples = []
    truncated = 0
    evaluated = 0
    for row in rows:
        sample, is_truncated = _sample(len(samples), row, coordinate is not None)
        samples.append(sample)
        truncated += is_truncated
        evaluated += row.get('eval_id') is not None
    metrics = _scalar_metrics(task['metrics'])
    primary_metric = 'score' if 'score' in metrics else sorted(metrics)[0]
    comparison = None
    if coordinate is not None:
        comparison_value, selected_model = coordinate
        comparison = {
            'model': selected_model,
            'benchmark': {
                'label': benchmark,
                'categories': [{
                    'id': 'function_call',
                    'label': 'Function Call'
                }],
                'evaluation_method': str(task['cot_mode']).lower(),
                'score_multiplier': 100.0,
            },
            'evaluation': {
                'prompt_profile': 'legacy-evalscope',
                'prompt_template': '',
                'precision': mapping['wkv_mode'],
            },
            'coordinates': [comparison_value],
            'samples': len(samples),
            'truncation_rate': truncated / len(samples) if samples else 0.0,
        }
    return {
        'schema_version': SCHEMA_VERSION,
        'task': _task_identity(task, mapping),
        'result_files': [],
        'task_config': {
            'old_task_id': int(task['task_id']),
            'old_score_id': int(task['score_id']),
            'evaluator': task['evaluator'],
            'cot_mode': task['cot_mode'],
            'config_path': task.get('config_path'),
            'git_hash': task.get('git_hash'),
        },
        'environment': {
            'framework': 'evalscope'
        },
        'sampling_config': task.get('sampling_config') or {},
        'primary_metric': primary_metric,
        'metrics': metrics,
        'diagnostics': {
            'legacy_metrics': task['metrics'],
            'source_counts': {
                'completions': len(samples),
                'evaluated': evaluated,
            },
        },
        'samples': samples,
        'comparison': comparison,
    }


def build_migration_bundle(
    source_rows: Iterable[Mapping[str, Any]],
    mappings: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], Iterable[dict[str, Any]]]:
    rows = iter(source_rows)
    models: list[Mapping[str, Any]] = []
    tasks: list[Mapping[str, Any]] = []
    first_sample: Mapping[str, Any] | None = None
    for row in rows:
        if row['kind'] == 'model':
            models.append(row)
        elif row['kind'] == 'task':
            tasks.append(row)
        elif row['kind'] == 'sample':
            first_sample = row
            break
    if not tasks:
        raise ValueError('legacy database contains no completed EvalScope scores')
    missing = sorted({str(task['model_name']) for task in tasks} - set(mappings))
    if missing:
        raise ValueError(f"model map is missing: {', '.join(missing)}")
    task_by_id = {int(task['task_id']): task for task in tasks}

    def publications() -> Iterable[dict[str, Any]]:
        sample_rows = rows if first_sample is None else chain([first_sample], rows)
        published = set()
        for task_id, task_rows in groupby(sample_rows, key=lambda row: int(row['task_id'])):
            task = task_by_id[task_id]
            published.add(task_id)
            publication = _task_publication(
                task,
                task_rows,
                models,
                mappings[str(task['model_name'])],
            )
            yield publication
            del publication
        for task_id, task in task_by_id.items():
            if task_id not in published:
                yield _task_publication(task, [], models, mappings[str(task['model_name'])])

    benchmarks = sorted({str(task['benchmark_name']) for task in tasks})
    campaign = {
        'schema_version': SCHEMA_VERSION,
        'run_key': '',
        'source': 'evalscope',
        'config_sha256': json_sha256([{
            'task_id': task['task_id'],
            'config_path': task.get('config_path'),
            'git_hash': task.get('git_hash'),
            'sampling_config': task.get('sampling_config'),
        } for task in tasks]),
        'registry_sha256': json_sha256([{
            'benchmark_id': task['benchmark_id'],
            'benchmark_name': task['benchmark_name'],
            'benchmark_split': task.get('benchmark_split'),
            'benchmark_samples': task.get('benchmark_samples'),
        } for task in tasks]),
        'contract_sha256': json_sha256(MIGRATION_CONTRACT),
        'configured_benchmarks': benchmarks,
        'resolved_benchmarks': benchmarks,
        'skipped_benchmarks': [],
        'expected_tasks': [_task_identity(task, mappings[str(task['model_name'])]) for task in tasks],
        'rerun_reason': None,
    }
    campaign['run_key'] = campaign_run_key(campaign)
    canonical_json(campaign)
    return campaign, publications()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-map', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    campaign, tasks = build_migration_bundle(read_source_rows(), load_model_mappings(args.model_map))
    if args.dry_run:
        task_count = 0
        sample_count = 0
        comparison_count = 0
        uncompressed_bytes = 0
        largest_task_bytes = 0
        for task in tasks:
            task_bytes = len(canonical_json(task))
            task_count += 1
            sample_count += len(task['samples'])
            comparison_count += task['comparison'] is not None
            uncompressed_bytes += task_bytes
            largest_task_bytes = max(largest_task_bytes, task_bytes)
            task = None
        result = {
            'run_key': campaign['run_key'],
            'tasks': task_count,
            'samples': sample_count,
            'comparison_tasks': comparison_count,
            'uncompressed_bytes': uncompressed_bytes,
            'largest_task_bytes': largest_task_bytes,
        }
    else:
        result = publish_bundle(
            campaign,
            tasks,
            os.environ['SCOREBOARD_API_BASE_URL'],
            os.environ['SCOREBOARD_PUBLICATION_TOKEN'],
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
