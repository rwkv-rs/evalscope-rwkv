from __future__ import annotations

import gzip
import hashlib
import json
from copy import deepcopy
from typing import Any

import pytest

from evalscope.utils.scoreboard import publish_benchmark_callback
from tools.scoreboard.migrate_legacy import (
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


def write_native_artifacts(root: Any, scores: list[float | None] | None = None) -> Any:
    model_dir = 'rwkv7-g1i-1.5b'
    benchmark = 'general_fc'
    (root / 'configs').mkdir()
    (root / 'reports' / model_dir).mkdir(parents=True)
    (root / 'predictions' / model_dir).mkdir(parents=True)
    (root / 'reviews' / model_dir).mkdir(parents=True)
    config = {
        'evalscope_version': '1.11.0',
        'eval_type': 'rwkv_openai_api',
        'generation_config': {'temperature': 0.0, 'max_tokens': 2048},
        'limit': None,
        'repeats': 1,
        'seed': 42,
        'resolved_benchmarks': {
            benchmark: {
                'dataset_id': 'evalscope/general_fc',
                'eval_split': 'test',
                'prompt_template': 'User: {question}\nAssistant:',
                'extra_params': {'use_cot': False},
            },
        },
        'evaluation_identity': {
            'benchmarks': {benchmark: {'evaluation_version': 'v4', 'fingerprint': 'sha256:abc'}},
        },
    }
    report = {
        'schema_version': 2,
        'dataset_pretty_name': 'Function Call',
        'primary_metric_identity': {'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}},
        'metrics': [{
            'identity': {'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}},
            'score': 0.75,
            'semantics': {'display_multiplier': 100.0},
        }],
        'execution_summary': {'requested': 1, 'succeeded': 1, 'errored': 0, 'incomplete': False},
    }
    prediction = {
        'index': 7,
        'model_output': {
            'choices': [{
                'message': {'content': '', 'tool_calls': [{'function': {'name': 'weather', 'arguments': '{}'}}]},
                'stop_reason': 'stop',
            }],
            'usage': {'output_tokens': 9},
            'time': 0.125,
        },
        'messages': [{'role': 'user', 'content': 'Call weather.'}, {'role': 'assistant', 'content': ''}],
    }
    review = {
        'index': 7,
        'target': {'name': 'weather'},
        'messages': prediction['messages'],
        'sample_score': {
            'score': {
                'value': {'accuracy': 1.0},
                'status': 'success',
                'extracted_prediction': {'name': 'weather'},
                'prediction': {'name': 'weather', 'arguments': {}},
                'explanation': None,
            },
            'group_id': 'weather-7',
            'generation_index': 0,
            'sample_metadata': {'case': 'weather'},
        },
    }
    metadata = {
        'weight_sha256': 'b' * 64,
        'weight_display_name': 'rwkv7-g1i-1.5b.pth',
        'wkv_mode': 'fp32io16',
        'comparison': {
            'model': {'label': 'RWKV G1I 1.5B', 'architecture': 'RWKV7', 'generation': 'G1I', 'parameters': '1.5B'},
            'coordinates': [{
                'comparison': {
                    'id': 'generation',
                    'label': 'G1H vs G1I',
                    'short_label': 'Generation',
                    'a_label': 'G1H',
                    'b_label': 'G1I',
                    'contract': 'Only the model generation changes.',
                },
                'parameter_group': {
                    'id': '1.5b',
                    'label': '1.5B',
                    'a_model': {'label': 'RWKV G1H 1.5B', 'architecture': 'RWKV7',
                                'generation': 'G1H', 'parameters': '1.5B'},
                    'b_model': {'label': 'RWKV G1I 1.5B', 'architecture': 'RWKV7',
                                'generation': 'G1I', 'parameters': '1.5B'},
                    'parameter_delta_percent': 0.0,
                    'comparable': True,
                },
                'arm': 'b',
            }],
        },
    }
    (root / 'configs' / 'task_config.yaml').write_text(json.dumps(config), encoding='utf-8')
    (root / 'reports' / model_dir / f'{benchmark}.json').write_text(json.dumps(report), encoding='utf-8')
    row_name = f'{benchmark}_default.jsonl'
    predictions = []
    reviews = []
    for index, value in enumerate(scores or [1.0], start=7):
        prediction_row = deepcopy(prediction)
        prediction_row['index'] = index
        review_row = deepcopy(review)
        review_row['index'] = index
        review_row['sample_score']['group_id'] = f'weather-{index}'
        if value is None:
            prediction_row['model_output']['choices'][0]['message'] = {'content': '', 'tool_calls': []}
            review_row['sample_score']['score']['prediction'] = ''
            review_row['sample_score']['score']['value'] = {}
        else:
            review_row['sample_score']['score']['value']['accuracy'] = value
        predictions.append(prediction_row)
        reviews.append(review_row)
    (root / 'predictions' / model_dir / row_name).write_text(
        ''.join(f'{json.dumps(row)}\n' for row in predictions), encoding='utf-8'
    )
    (root / 'reviews' / model_dir / row_name).write_text(
        ''.join(f'{json.dumps(row)}\n' for row in reviews), encoding='utf-8'
    )
    metadata_path = root / 'scoreboard.json'
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    return model_dir, benchmark, metadata_path


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
        'metrics': {
            'score': 0.5,
            'imported_evals': 2,
            'not-a-number': 'ignored',
        },
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


def test_native_callback_publishes_saved_score_io_and_decoding_config(tmp_path: Any, monkeypatch: Any) -> None:
    from evalscope.utils import scoreboard

    scores = [1.0] * 21 + [0.0] * 21 + [0.5] * 21 + [None] * 21
    model_id, benchmark, metadata_path = write_native_artifacts(tmp_path, scores)
    campaign_id = '00000000-0000-0000-0000-000000000001'
    requests = []
    responses = iter([
        Response(201, {'campaign_id': campaign_id, 'action': 'created', 'status': 'incomplete'}),
        Response(201, {'evaluation_id': 'evaluation-id', 'action': 'created'}),
        Response(200, {'campaign_id': campaign_id, 'status': 'complete', 'task_count': 1}),
    ])

    def opener(request: Any, *, timeout: int) -> Response:
        requests.append(request)
        assert timeout == 60
        return next(responses)

    monkeypatch.setenv('SCOREBOARD_API_BASE_URL', 'http://scoreboard.test')
    monkeypatch.setenv('SCOREBOARD_PUBLICATION_TOKEN', 'publication-token')
    monkeypatch.setattr(scoreboard, 'urlopen', opener)
    receipt = publish_benchmark_callback(str(tmp_path), model_id, benchmark, str(metadata_path))

    assert receipt['task']['evaluation_id'] == 'evaluation-id'
    assert [request.get_method() for request in requests] == ['POST', 'PUT', 'POST']
    campaign = json.loads(gzip.decompress(requests[0].data))
    publication = json.loads(gzip.decompress(requests[1].data))
    assert campaign['expected_tasks'] == [publication['task']]
    assert publication['campaign_id'] == campaign_id
    assert publication['primary_metric'] == 'accuracy'
    assert publication['metrics'] == {'accuracy': 0.75}
    assert publication['sampling_config'] == {'max_tokens': 2048, 'temperature': 0.0}
    assert publication['comparison']['samples'] == 80
    assert publication['comparison']['benchmark']['score_multiplier'] == 100.0
    assert publication['diagnostics']['samples_total'] == 84
    assert publication['diagnostics']['samples_uploaded'] == 80
    assert publication['result_files'] == []
    assert [sample['sample_index'] for sample in publication['samples']] == list(range(80))
    outcomes = [sample['answer']['outcome'] for sample in publication['samples']]
    assert outcomes.count('correct') == 20
    assert outcomes.count('incorrect') == 20
    assert outcomes.count('undetermined') == 20
    assert outcomes.count('unanswered') == 20
    sample = publication['samples'][0]
    assert sample['answer']['outcome'] == 'correct'
    assert sample['answer']['ground_truth'] == '{"name":"weather"}'
    assert sample['answer']['raw_completion'] == '[{"function":{"arguments":"{}","name":"weather"}}]'
    assert sample['answer']['assembled_prompt'] == '[{"content":"Call weather.","role":"user"}]'
    assert sample['answer']['generated_tokens'] == 9
    assert sample['answer']['latency_ms'] == 125.0
    assert sample['model_response']['tool_calls'][0]['function']['name'] == 'weather'


def test_native_callback_checks_scoreboard_request_limits(tmp_path: Any, monkeypatch: Any) -> None:
    from evalscope.utils import scoreboard

    model_id, benchmark, metadata_path = write_native_artifacts(tmp_path)
    monkeypatch.setenv('SCOREBOARD_API_BASE_URL', 'http://scoreboard.test')
    monkeypatch.setenv('SCOREBOARD_PUBLICATION_TOKEN', 'publication-token')
    monkeypatch.setattr(scoreboard, 'MAX_UNCOMPRESSED_BYTES', 1)

    with pytest.raises(ValueError, match='256 MiB raw limit'):
        publish_benchmark_callback(str(tmp_path), model_id, benchmark, str(metadata_path))


def test_migration_upload_checks_scoreboard_request_limits(monkeypatch: Any) -> None:
    from tools.scoreboard import migrate_legacy

    monkeypatch.setattr(migrate_legacy, 'MAX_UNCOMPRESSED_BYTES', 1)
    with pytest.raises(ValueError, match='256 MiB raw limit'):
        migrate_legacy.send(
            'http://scoreboard.test', 'publication-token', 'POST', '/campaigns', 'campaign:key', CAMPAIGN
        )


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
    assert publication['metrics'] == {'avg@1': 0.5}
    assert publication['primary_metric'] == 'avg@1'
    assert publication['diagnostics']['source_counts'] == {'completions': 2, 'evaluated': 1, 'uploaded': 2}
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


def test_migration_limits_evidence_per_outcome() -> None:
    rows = source_rows()
    correct = rows[-2]
    correct_samples = [{
        **correct,
        'completion_id': 1000 + index,
        'sample_index': index,
        'source_index': index,
        'eval_id': 2000 + index,
    } for index in range(21)]
    incorrect_samples = [{
        **correct,
        'completion_id': 3000 + index,
        'sample_index': 21 + index,
        'source_index': 21 + index,
        'eval_id': 4000 + index,
        'answer': 'wrong',
        'is_passed': False,
    } for index in range(21)]
    unanswered_samples = [{
        **correct,
        'completion_id': 5000 + index,
        'sample_index': 42 + index,
        'source_index': 42 + index,
        'eval_id': None,
        'answer': None,
        'is_passed': None,
    } for index in range(21)]
    samples = [*correct_samples, *incorrect_samples, *unanswered_samples]
    _campaign, publications = build_migration_bundle([*rows[:3], *samples], MODEL_MAP)
    publication = next(iter(publications))

    assert [sample['sample_index'] for sample in publication['samples']] == list(range(60))
    outcomes = [sample['answer']['outcome'] for sample in publication['samples']]
    assert outcomes.count('correct') == 20
    assert outcomes.count('incorrect') == 20
    assert outcomes.count('unanswered') == 20
    assert publication['diagnostics']['source_counts'] == {
        'completions': 63,
        'evaluated': 42,
        'uploaded': 60,
    }


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
