from __future__ import annotations

import gzip
import hashlib
import json
from typing import Any

import pytest

from scripts.upload_score import (
    SOURCE_QUERY,
    build_migration_bundle,
    canonical_json,
    load_model_mappings,
    publish_bundle,
)

CAMPAIGN = {
    'schema_version': 'scoreboard-v1',
    'run_key': '1' * 64,
    'source': 'evalscope',
    'expected_tasks': [],
}
TASK = {
    'schema_version': 'scoreboard-v1',
    'task': {'identity': 'weight:mode:task/name'},
    'metrics': {'score': 0.75},
    'samples': [{'sample_index': 0, 'model_response': {'text': 'answer'}}],
}
WEIGHT_SHA256 = 'a' * 64
MODEL_MAP = {
    'rwkv7-g1h-1.5b': {
        'weight_sha256': WEIGHT_SHA256,
        'weight_display_name': 'rwkv7-g1h-1.5b.pth',
        'wkv_mode': 'fp32io16',
    },
}


def source_rows() -> list[dict[str, Any]]:
    models = [
        {
            'kind': 'model',
            'model_id': 1,
            'model_name': 'rwkv7-g1h-1.5b',
            'arch_version': 'rwkv7',
            'data_version': 'g1h',
            'num_params': '1_5b',
        },
        {
            'kind': 'model',
            'model_id': 2,
            'model_name': 'rwkv7-g1i-1.5b',
            'arch_version': 'rwkv7',
            'data_version': 'g1i',
            'num_params': '1_5b',
        },
    ]
    task = {
        'kind': 'task',
        'task_id': 42,
        'score_id': 7,
        'model_id': 1,
        'model_name': 'rwkv7-g1h-1.5b',
        'arch_version': 'rwkv7',
        'data_version': 'g1h',
        'num_params': '1_5b',
        'benchmark_id': 3,
        'benchmark_name': 'bfcl_v4',
        'benchmark_split': 'test',
        'benchmark_samples': 2,
        'evaluator': 'evalscope-native',
        'config_path': 'configs/bfcl.py',
        'git_hash': 'abc123',
        'sampling_config': {'temperature': 0},
        'cot_mode': 'none',
        'metrics': {'score': 0.5, 'not-a-number': 'ignored'},
    }
    sample = {
        'kind': 'sample',
        'task_id': 42,
        'completion_id': 100,
        'sample_index': 5,
        'repeat_index': 0,
        'pass_index': 0,
        'messages': [{'role': 'user', 'content': 'Call the weather tool.'}],
        'sample_score': {
            'group_id': 'weather-5',
            'score': {
                'prediction': {'name': 'weather', 'arguments': {'city': 'Shanghai'}},
                'extracted_prediction': 'weather(city="Shanghai")',
                'target': 'weather(city="Shanghai")',
                'value': {'accuracy': 1.0},
            },
        },
        'model_output': {
            'choices': [{
                'message': {
                    'content': 'weather(city="Shanghai")',
                    'perf_metrics': {'output_tokens': 12, 'latency': 0.25},
                },
                'stop_reason': 'stop',
            }],
            'usage': {},
        },
        'context_audit': {},
        'source_index': 5,
        'subset': 'test',
        'eval_id': 200,
        'answer': 'weather(city="Shanghai")',
        'reference_answer': 'weather(city="Shanghai")',
        'is_passed': True,
        'fail_reason': None,
    }
    unanswered = {
        **sample,
        'completion_id': 101,
        'sample_index': 6,
        'source_index': 6,
        'eval_id': None,
        'answer': None,
        'is_passed': None,
        'fail_reason': 'no_answer',
    }
    return [*models, task, sample, unanswered]


class Response:
    def __init__(self, status: int, value: dict[str, Any]) -> None:
        self.status = status
        self.body = canonical_json(value)

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_publish_bundle_uses_scoreboard_http_lifecycle() -> None:
    requests = []
    responses = iter([
        Response(201, {
            'campaign_id': 'campaign-id',
            'action': 'created',
            'status': 'incomplete',
        }),
        Response(201, {'evaluation_id': 'evaluation-id', 'action': 'created'}),
        Response(200, {'campaign_id': 'campaign-id', 'status': 'complete'}),
    ])

    def opener(request: Any, *, timeout: int) -> Response:
        requests.append(request)
        assert timeout == 60
        return next(responses)

    receipt = publish_bundle(
        CAMPAIGN,
        [TASK],
        'http://scoreboard.test/',
        'publication-token',
        opener=opener,
    )

    assert receipt['campaign']['http_status'] == 201
    assert receipt['tasks'][0]['evaluation_id'] == 'evaluation-id'
    assert receipt['finalize']['status'] == 'complete'
    assert [request.get_method() for request in requests] == ['POST', 'PUT', 'POST']
    assert requests[0].full_url == 'http://scoreboard.test/api/v1/evaluation-campaigns'
    assert requests[1].full_url == (
        'http://scoreboard.test/api/v1/evaluation-campaigns/campaign-id/tasks/weight%3Amode%3Atask%2Fname'
    )
    assert requests[2].full_url.endswith('/campaign-id/finalize')

    campaign_body = gzip.decompress(requests[0].data)
    task_body = gzip.decompress(requests[1].data)
    assert json.loads(campaign_body) == CAMPAIGN
    assert json.loads(task_body) == {**TASK, 'campaign_id': 'campaign-id'}
    assert requests[0].get_header('Idempotency-key') == f"campaign:{CAMPAIGN['run_key']}"
    assert requests[1].get_header('Idempotency-key') == f'publish:{hashlib.sha256(task_body).hexdigest()}'
    assert requests[2].get_header('Idempotency-key') == 'finalize:campaign-id'
    assert all(request.get_header('Authorization') == 'Bearer publication-token' for request in requests)


def test_canonical_json_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError):
        canonical_json({'score': float('nan')})


def test_publish_bundle_skips_a_complete_campaign() -> None:
    requests = []

    def opener(request: Any, *, timeout: int) -> Response:
        requests.append(request)
        return Response(200, {
            'campaign_id': 'campaign-id',
            'action': 'unchanged',
            'status': 'complete',
        })

    receipt = publish_bundle(
        CAMPAIGN,
        [TASK],
        'http://scoreboard.test',
        'publication-token',
        opener=opener,
    )

    assert receipt['campaign']['status'] == 'complete'
    assert receipt['tasks'] == []
    assert receipt['finalize'] is None
    assert [request.get_method() for request in requests] == ['POST']


def test_build_migration_bundle_preserves_scores_outputs_and_answers() -> None:
    campaign, publications = build_migration_bundle(source_rows(), MODEL_MAP)
    publication = next(iter(publications))

    assert campaign['source'] == 'evalscope'
    assert campaign['configured_benchmarks'] == ['bfcl_v4']
    assert campaign['expected_tasks'] == [publication['task']]
    assert len(campaign['run_key']) == 64
    assert publication['metrics'] == {'score': 0.5}
    assert publication['primary_metric'] == 'score'
    assert publication['diagnostics']['source_counts'] == {'completions': 2, 'evaluated': 1}
    assert publication['comparison']['model']['generation'] == 'G1H'
    coordinate = publication['comparison']['coordinates'][0]
    assert coordinate['arm'] == 'a'
    assert coordinate['parameter_group']['a_model']['generation'] == 'G1H'
    assert coordinate['parameter_group']['b_model']['generation'] == 'G1I'

    correct, unanswered = publication['samples']
    assert correct['document_index'] == 5
    assert correct['metrics'] == {'accuracy': 1.0, 'has_eval_record': True, 'is_passed': True}
    assert correct['model_response']['text'] == '{"arguments":{"city":"Shanghai"},"name":"weather"}'
    assert correct['model_response']['stop_reason'] == 'stop'
    assert correct['answer'] == {
        'outcome': 'correct',
        'problem_id': 'weather-5',
        'repeat_id': 0,
        'ground_truth': 'weather(city="Shanghai")',
        'extracted_answer': 'weather(city="Shanghai")',
        'assembled_prompt': '[{"content":"Call the weather tool.","role":"user"}]',
        'raw_completion': '{"arguments":{"city":"Shanghai"},"name":"weather"}',
        'fail_reason': None,
        'generated_tokens': 12,
        'latency_ms': 250.0,
    }
    assert unanswered['answer']['outcome'] == 'unanswered'
    assert unanswered['answer']['fail_reason'] == 'no_answer'


def test_bundle_is_streamed_and_requires_every_model_mapping() -> None:
    consumed = []

    def rows() -> Any:
        for row in source_rows():
            consumed.append(row['kind'])
            yield row

    _campaign, publications = build_migration_bundle(rows(), MODEL_MAP)
    assert consumed == ['model', 'model', 'task', 'sample']
    assert len(next(iter(publications))['samples']) == 2
    assert consumed == ['model', 'model', 'task', 'sample', 'sample']

    with pytest.raises(ValueError, match='model map is missing: rwkv7-g1h-1.5b'):
        build_migration_bundle(source_rows(), {})


def test_load_model_mappings_validates_publication_identity(tmp_path: Any) -> None:
    model_map = tmp_path / 'models.json'
    model_map.write_text(json.dumps(MODEL_MAP), encoding='utf-8')
    assert load_model_mappings(model_map) == MODEL_MAP

    model_map.write_text(json.dumps({'model': {'weight_sha256': 'bad'}}), encoding='utf-8')
    with pytest.raises(ValueError, match='lowercase SHA-256'):
        load_model_mappings(model_map)


def test_source_query_is_read_only_and_selects_only_evalscope() -> None:
    normalized = ' '.join(SOURCE_QUERY.lower().split())
    assert 'repeatable read read only' in normalized
    assert "lower(t.evaluator) like 'evalscope%'" in normalized
    assert 'insert ' not in normalized
    assert 'update ' not in normalized
    assert 'delete ' not in normalized
