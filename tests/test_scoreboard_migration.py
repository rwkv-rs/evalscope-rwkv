from __future__ import annotations

import asyncio
import getpass
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterator

import asyncpg
import pytest

from scripts.migrate_helicopter_scoreboard import (
    MigrationConflictError,
    MigrationError,
    _configure_connection,
    migrate_legacy_scoreboard,
    parse_model_mappings,
    split_scalar_metrics,
)

TARGET_SCHEMA_SQL = '''
CREATE TABLE evaluation_schema_metadata (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    contract_version integer NOT NULL
);
INSERT INTO evaluation_schema_metadata VALUES (true, 4);
CREATE TABLE evaluation_campaign (
    id uuid PRIMARY KEY,
    run_key text NOT NULL UNIQUE,
    status text NOT NULL,
    source text NOT NULL,
    config_sha256 text NOT NULL,
    registry_sha256 text NOT NULL,
    contract_sha256 text NOT NULL,
    configured_benchmarks jsonb NOT NULL,
    resolved_benchmarks jsonb NOT NULL,
    skipped_benchmarks jsonb NOT NULL,
    expected_tasks jsonb NOT NULL,
    rerun_reason text,
    publisher text NOT NULL,
    created_at timestamptz NOT NULL,
    completed_at timestamptz
);
CREATE TABLE evaluation_task (
    id uuid PRIMARY KEY,
    campaign_id uuid NOT NULL REFERENCES evaluation_campaign(id) ON DELETE CASCADE,
    task_identity text NOT NULL,
    content_sha256 text NOT NULL,
    task jsonb NOT NULL,
    result_files jsonb NOT NULL,
    task_config jsonb NOT NULL,
    environment jsonb NOT NULL,
    sampling_config jsonb NOT NULL,
    primary_metric text NOT NULL,
    metrics jsonb NOT NULL,
    diagnostics jsonb NOT NULL,
    created_at timestamptz NOT NULL,
    UNIQUE (campaign_id, task_identity)
);
CREATE TABLE evaluation_sample (
    evaluation_id uuid NOT NULL REFERENCES evaluation_task(id) ON DELETE CASCADE,
    sample_index integer NOT NULL,
    document_index integer NOT NULL,
    document jsonb NOT NULL,
    metrics jsonb NOT NULL,
    model_response jsonb NOT NULL,
    PRIMARY KEY (evaluation_id, sample_index)
);
'''


def _postgres_config(*, database: str = 'postgres') -> dict[str, object]:
    return {
        'host': os.environ.get('PGHOST') or '/var/run/postgresql',
        'port': int(os.environ.get('PGPORT') or 5432),
        'user': os.environ.get('PGUSER') or getpass.getuser(),
        'password': os.environ.get('PGPASSWORD'),
        'database': database,
    }


@pytest.fixture()
def migration_databases() -> Iterator[tuple[asyncio.AbstractEventLoop, asyncpg.Connection, asyncpg.Connection]]:
    if os.environ.get('SCOREBOARD_MIGRATION_TEST_POSTGRES') != '1':
        pytest.skip('set SCOREBOARD_MIGRATION_TEST_POSTGRES=1 to run PostgreSQL migration integration tests')
    source_name = f'helicopter_migration_source_{uuid.uuid4().hex[:12]}'
    target_name = f'helicopter_migration_target_{uuid.uuid4().hex[:12]}'
    loop = asyncio.new_event_loop()

    async def setup() -> tuple[asyncpg.Connection, asyncpg.Connection, asyncpg.Connection]:
        admin = await asyncpg.connect(**_postgres_config())
        await admin.execute(f'CREATE DATABASE "{source_name}"')
        await admin.execute(f'CREATE DATABASE "{target_name}"')
        source = await asyncpg.connect(**_postgres_config(database=source_name))
        target = await asyncpg.connect(**_postgres_config(database=target_name))
        await _configure_connection(target)
        await target.execute(TARGET_SCHEMA_SQL)
        return admin, source, target

    admin, source, target = loop.run_until_complete(setup())
    try:
        yield loop, source, target
    finally:
        async def cleanup() -> None:
            await source.close()
            await target.close()
            for name in (source_name, target_name):
                await admin.execute(
                    '''
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = $1 AND pid <> pg_backend_pid()
                    ''',
                    name,
                )
                await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
            await admin.close()

        loop.run_until_complete(cleanup())
        loop.close()


def test_model_mapping_requires_real_weight_metadata_and_splits_metrics() -> None:
    with pytest.raises(MigrationError, match='64-hex weight_sha256'):
        parse_model_mappings({
            'rwkv-legacy': {
                'weight_sha256': 'model-name-is-not-a-content-hash',
                'wkv_mode': 'fp32io16',
            }
        })
    with pytest.raises(MigrationError, match='requires wkv_mode'):
        parse_model_mappings({'rwkv-legacy': {'weight_sha256': 'a' * 64}})

    scalars, diagnostics = split_scalar_metrics({
        'accuracy': 0.75,
        'samples': 4,
        'latency': {'p50': 0.1},
        'tokens': None,
        'invalid': float('nan'),
    })
    assert scalars == {'accuracy': 0.75, 'samples': 4.0}
    assert diagnostics['legacy_metrics']['latency'] == {'p50': 0.1}
    assert diagnostics['legacy_metrics']['tokens'] is None
    assert diagnostics['legacy_metrics']['invalid'] == {'legacy_non_finite_number': 'nan'}


async def _prepare_legacy_database(connection: asyncpg.Connection) -> None:
    await _configure_connection(connection)
    await connection.execute(
        '''
        CREATE TABLE model (
            model_id integer PRIMARY KEY,
            data_version text NOT NULL,
            arch_version text NOT NULL,
            num_params text NOT NULL,
            model_name text NOT NULL
        );
        CREATE TABLE benchmark (
            benchmark_id integer PRIMARY KEY,
            benchmark_name text NOT NULL,
            benchmark_split text NOT NULL,
            url text,
            status text NOT NULL,
            num_samples integer NOT NULL
        );
        CREATE TABLE benchmark_catalog (
            catalog_id integer PRIMARY KEY,
            benchmark_name text NOT NULL,
            benchmark_split text NOT NULL,
            field text NOT NULL,
            source text NOT NULL,
            source_family text NOT NULL,
            target_kind text NOT NULL,
            run_status text NOT NULL,
            scope text NOT NULL,
            metadata jsonb,
            created_at timestamptz NOT NULL,
            updated_at timestamptz NOT NULL
        );
        CREATE TABLE task (
            task_id integer PRIMARY KEY,
            config_path text,
            evaluator text NOT NULL,
            is_param_search boolean NOT NULL,
            is_tmp boolean NOT NULL,
            created_at timestamptz NOT NULL,
            status text NOT NULL,
            git_hash text NOT NULL,
            model_id integer NOT NULL,
            benchmark_id integer NOT NULL,
            "desc" text,
            sampling_config jsonb,
            log_path text NOT NULL
        );
        CREATE TABLE scores (
            score_id integer PRIMARY KEY,
            task_id integer NOT NULL UNIQUE,
            cot_mode text NOT NULL,
            metrics jsonb NOT NULL,
            created_at timestamptz NOT NULL
        );
        CREATE TABLE completions (
            completions_id integer PRIMARY KEY,
            task_id integer NOT NULL,
            context jsonb NOT NULL,
            sample_index integer NOT NULL,
            avg_repeat_index integer NOT NULL,
            pass_index integer NOT NULL,
            created_at timestamptz NOT NULL,
            status text NOT NULL
        );
        CREATE TABLE eval (
            eval_id integer PRIMARY KEY,
            completions_id integer NOT NULL UNIQUE,
            answer text NOT NULL,
            ref_answer text NOT NULL,
            is_passed boolean NOT NULL,
            fail_reason text NOT NULL,
            created_at timestamptz NOT NULL
        );
        CREATE TABLE checker (
            checker_id integer PRIMARY KEY,
            completions_id integer NOT NULL UNIQUE,
            answer_correct boolean NOT NULL,
            instruction_following_error boolean NOT NULL,
            world_knowledge_error boolean NOT NULL,
            math_error boolean NOT NULL,
            reasoning_logic_error boolean NOT NULL,
            thought_contains_correct_answer boolean NOT NULL,
            needs_human_review boolean NOT NULL,
            reason text NOT NULL,
            created_at timestamptz NOT NULL
        );
        '''
    )
    base = datetime(2026, 7, 24, 8, 0, tzinfo=timezone.utc)
    await connection.executemany(
        'INSERT INTO model VALUES ($1, $2, $3, $4, $5)',
        [
            (1, 'data-a', 'rwkv7', '2.9B', 'rwkv-model-a'),
            (2, 'data-b', 'rwkv7', '7.2B', 'rwkv-model-b'),
        ],
    )
    await connection.executemany(
        'INSERT INTO benchmark VALUES ($1, $2, $3, $4, $5, $6)',
        [
            (1, 'math_500', '', 'https://example.test/math', 'ready', 500),
            (2, 'arc', 'challenge', 'https://example.test/arc', 'ready', 1172),
            (3, 'bfcl_simple_python', '', None, 'ready', 2),
        ],
    )
    await connection.executemany(
        'INSERT INTO benchmark_catalog VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)',
        [
            (
                1,
                'math_500',
                '',
                'math',
                'lighteval',
                'reasoning',
                'generation',
                'ready',
                'scoreboard',
                {'languages': ['english']},
                base,
                base,
            ),
            (
                2,
                'arc',
                'challenge',
                'knowledge',
                'lighteval',
                'multiple-choice',
                'choice',
                'ready',
                'scoreboard',
                None,
                base,
                base,
            ),
        ],
    )
    await connection.executemany(
        '''
        INSERT INTO task (
            task_id, config_path, evaluator, is_param_search, is_tmp,
            created_at, status, git_hash, model_id, benchmark_id, "desc",
            sampling_config, log_path
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        ''',
        [
            (
                1,
                'config-old.toml',
                'lighteval',
                False,
                False,
                base,
                'Completed',
                'git-old',
                1,
                1,
                'old math run',
                {'temperature': 0.8},
                '/legacy/task-1.log',
            ),
            (
                2,
                'config-new.toml',
                'lighteval',
                False,
                False,
                base + timedelta(days=1),
                'Completed',
                'git-new',
                1,
                1,
                'new math run',
                {'temperature': 0.7},
                '/legacy/task-2.log',
            ),
            (
                3,
                'config-arc.toml',
                'lighteval',
                False,
                False,
                base + timedelta(days=2),
                'Completed',
                'git-arc',
                2,
                2,
                'arc run',
                {'max_tokens': 32},
                '/legacy/task-3.log',
            ),
            (
                4,
                'config-bfcl.toml',
                'function_calling',
                False,
                False,
                base + timedelta(days=3),
                'Completed',
                'git-bfcl',
                1,
                3,
                'function calling run',
                {},
                '/legacy/task-4.log',
            ),
            (
                5,
                'config-running.toml',
                'lighteval',
                False,
                False,
                base + timedelta(days=4),
                'Running',
                'git-running',
                2,
                1,
                'unfinished',
                {},
                '/legacy/task-5.log',
            ),
        ],
    )
    await connection.executemany(
        'INSERT INTO scores VALUES ($1, $2, $3, $4, $5)',
        [
            (1, 1, 'old-cot', {'pass@1': 0.25, 'pass@1_stderr': 0.1}, base),
            (2, 2, 'new-cot', {'pass@1': 0.75, 'pass@1_stderr': 0.1}, base + timedelta(days=1)),
            (3, 3, 'no-cot', {'acc': 0.5, 'acc_stderr': 0.02}, base + timedelta(days=2)),
            (
                4,
                4,
                'tool',
                {
                    'samples': 2,
                    'accuracy': 0.5,
                    'error_rate': 0.5,
                    'latency': {'p50': 0.12},
                    'total_tokens': None,
                },
                base + timedelta(days=3),
            ),
        ],
    )
    await connection.executemany(
        'INSERT INTO completions VALUES ($1, $2, $3, $4, $5, $6, $7, $8)',
        [
            (101, 1, {'agent_result': 'old'}, 0, 0, 0, base, 'Completed'),
            (201, 2, {'agent_result': 'first'}, 2, 0, 0, base + timedelta(days=1), 'Completed'),
            (202, 2, {'agent_result': 'second'}, 2, 1, 0, base + timedelta(days=1, seconds=1), 'Completed'),
            (401, 4, {'agent_result': {'tool': 'weather'}}, 0, 0, 0, base + timedelta(days=3), 'Completed'),
        ],
    )
    await connection.executemany(
        'INSERT INTO eval VALUES ($1, $2, $3, $4, $5, $6, $7)',
        [
            (1001, 101, 'old', 'ref', False, 'wrong', base),
            (2001, 201, 'first', 'ref', True, '', base + timedelta(days=1)),
            (2002, 202, 'second', 'ref', False, 'wrong', base + timedelta(days=1, seconds=1)),
            (4001, 401, 'weather', 'weather', True, '', base + timedelta(days=3)),
        ],
    )
    await connection.execute(
        '''
        INSERT INTO checker VALUES (
            1, 202, false, false, false, true, false, false, true,
            'arithmetic mismatch', $1
        )
        ''',
        base + timedelta(days=1, seconds=1),
    )


def test_migration_dry_run_apply_idempotency_and_conflicts(
    migration_databases: tuple[asyncio.AbstractEventLoop, asyncpg.Connection, asyncpg.Connection],
) -> None:
    loop, source, target = migration_databases
    loop.run_until_complete(_assert_migration(source, target))


async def _assert_migration(source: asyncpg.Connection, target: asyncpg.Connection) -> None:
    await _prepare_legacy_database(source)
    mappings = parse_model_mappings({
        'rwkv-model-a': {
            'weight_sha256': 'a' * 64,
            'weight_display_name': 'rwkv-model-a.pth',
            'wkv_mode': 'fp32io16',
            'wkv_mode_source': 'historical-config-inference',
        },
        'rwkv-model-b': {
            'weight_sha256': 'b' * 64,
            'weight_display_name': 'rwkv-model-b.pth',
            'wkv_mode': 'fp32io16',
            'wkv_mode_source': 'historical-config-inference',
        },
    })

    dry_run = await migrate_legacy_scoreboard(source, target, mappings)
    assert dry_run.mode == 'dry-run'
    assert dry_run.source_task_count == 5
    assert dry_run.selected_score_count == 4
    assert dry_run.excluded_task_count == 1
    assert dry_run.planned_campaign_count == 3
    assert dry_run.snapshot_campaign_count == 2
    assert dry_run.historical_campaign_count == 1
    assert dry_run.planned_task_count == 4
    assert dry_run.planned_sample_count == 3
    assert dry_run.planned_attempt_count == 4
    assert dry_run.campaigns_to_create == 3
    assert await target.fetchval('SELECT count(*) FROM evaluation_campaign') == 0

    applied = await migrate_legacy_scoreboard(source, target, mappings, apply=True)
    assert applied.mode == 'apply'
    assert applied.campaigns_to_create == 3
    assert applied.tasks_to_create == 4
    assert await target.fetchval('SELECT count(*) FROM evaluation_campaign') == 3
    assert await target.fetchval('SELECT count(*) FROM evaluation_task') == 4
    assert await target.fetchval('SELECT count(*) FROM evaluation_sample') == 3
    assert await target.fetchval(
        '''
        SELECT jsonb_array_length(expected_tasks)
        FROM evaluation_campaign
        WHERE source = 'lighteval' AND rerun_reason IS NULL
        '''
    ) == 2
    assert await target.fetchval(
        'SELECT count(*) FROM evaluation_campaign WHERE completed_at = $1',
        datetime(2026, 7, 24, 8, 0, tzinfo=timezone.utc),
    ) == 1

    function_result = await target.fetchrow(
        '''
        SELECT id, primary_metric, metrics, diagnostics
        FROM evaluation_task
        WHERE diagnostics -> 'legacy' ->> 'score_id' = '4'
        '''
    )
    assert function_result is not None
    assert function_result['primary_metric'] == 'accuracy'
    assert function_result['metrics']['accuracy'] == 0.5
    assert function_result['diagnostics']['legacy']['legacy_metrics']['latency'] == {'p50': 0.12}
    assert function_result['diagnostics']['metadata_sources']['wkv_mode'] == 'historical-config-inference'

    latest_math = await target.fetchrow(
        '''
        SELECT id
        FROM evaluation_task
        WHERE diagnostics -> 'legacy' ->> 'score_id' = '2'
        '''
    )
    assert latest_math is not None
    sample = await target.fetchrow(
        '''
        SELECT sample_index, document_index, metrics, model_response
        FROM evaluation_sample
        WHERE evaluation_id = $1
        ''',
        latest_math['id'],
    )
    assert sample is not None
    assert sample['sample_index'] == 0
    assert sample['document_index'] == 2
    assert len(sample['model_response']['attempts']) == 2
    assert sample['metrics']['legacy_pass_rate'] == 0.5

    repeated = await migrate_legacy_scoreboard(source, target, mappings, apply=True)
    assert repeated.campaigns_to_create == 0
    assert repeated.campaigns_reused == 3
    assert repeated.tasks_reused == 4

    await target.execute(
        '''
        UPDATE evaluation_task
        SET metrics = metrics || '{"accuracy": 0.25}'::jsonb
        WHERE diagnostics -> 'legacy' ->> 'score_id' = '4'
        '''
    )
    with pytest.raises(MigrationConflictError, match='differs in: metrics'):
        await migrate_legacy_scoreboard(source, target, mappings)
