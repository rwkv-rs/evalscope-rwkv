"""Publish one completed native EvalScope benchmark to Scoreboard."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen

import yaml

SCHEMA_VERSION = 'scoreboard-v1'
MAX_COMPRESSED_BYTES, MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024, 256 * 1024 * 1024
MAX_SAMPLES_PER_OUTCOME = 20
CONTRACT = {
    'producer': 'evalscope',
    'artifact_schema': 3,
    'publication_schema': SCHEMA_VERSION,
    'max_samples_per_outcome': MAX_SAMPLES_PER_OUTCOME,
}


def _json(value: Any) -> bytes:
    options = {'ensure_ascii': False, 'allow_nan': False, 'sort_keys': True, 'separators': (',', ':')}
    return json.dumps(value, **options).encode('utf-8')


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _send(base_url, token, method, path, idempotency_key, payload) -> tuple[int, dict[str, Any]]:
    headers = {'Authorization': f'Bearer {token}', 'Idempotency-Key': idempotency_key}
    compressed = None
    if payload is not None:
        body = _json(payload)
        compressed = gzip.compress(body)
        if len(body) > MAX_UNCOMPRESSED_BYTES or len(compressed) > MAX_COMPRESSED_BYTES:
            raise ValueError('Scoreboard publication exceeds the 64 MiB compressed or 256 MiB raw limit')
        headers.update({'Content-Type': 'application/json', 'Content-Encoding': 'gzip'})
    request = Request(f"{base_url.rstrip('/')}{path}", data=compressed, method=method, headers=headers)
    with urlopen(request, timeout=60) as response:
        return response.status, json.loads(response.read().decode('utf-8'))


def _text(value: Any) -> str:
    return '' if value is None else value if isinstance(value, str) else _json(value).decode('utf-8')


def _metric_name(identity: Mapping[str, Any]) -> str:
    name = str(identity['name'])
    dimensions = identity.get('dimensions') or {}
    if dimensions:
        suffix = ','.join(f'{key}={dimensions[key]}' for key in sorted(dimensions))
        return f'{name}[{suffix}]'
    return name


def _report_metrics(report: Mapping[str, Any]) -> tuple[dict[str, float], str, str, float]:
    primary_identity = report['primary_metric_identity']
    primary_metric = _metric_name(primary_identity)
    metrics: dict[str, float] = {}
    multiplier = 1.0
    for item in report['metrics']:
        score = item.get('score')
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            metrics[_metric_name(item['identity'])] = float(score)
        if item['identity'] == primary_identity:
            multiplier = float((item.get('semantics') or {}).get('display_multiplier') or 1.0)
    if primary_metric not in metrics:
        raise ValueError(f'completed report has no finite primary metric: {primary_metric}')
    return metrics, primary_metric, str(primary_identity['name']), multiplier


def _sample(sample_index, subset, review, prediction, metric_name) -> tuple[dict[str, Any], bool]:
    sample_score = review['sample_score']
    score = sample_score['score']
    output = prediction.get('model_output') or {}
    choices = output.get('choices') or []
    choice = choices[0] if choices else {}
    message = choice.get('message') or {}
    raw_value = message.get('tool_calls') or message.get('content')
    raw_completion = _text(score.get('prediction') if raw_value is None else raw_value)
    score_name = score.get('main_score_name') or metric_name
    score_value = (score.get('value') or {}).get(score_name) if score.get('status') == 'success' else None
    outcome = 'unanswered' if not raw_completion.strip() else {
        1: 'correct',
        0: 'incorrect'
    }.get(score_value, 'undetermined')
    messages = list(review.get('messages') or prediction.get('messages') or [])
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    elapsed = output.get('time')
    latency_ms = float(elapsed) * 1000 if isinstance(elapsed, (int, float)) and elapsed >= 0 else None
    fail_reason = _text(score.get('explanation') or output.get('error')).strip()[:2000] or None
    group_id = sample_score.get('group_id')
    stop_reason = str(choice.get('stop_reason') or 'unknown')
    return {
        'sample_index': sample_index,
        'document_index': int(review['index']),
        'document': {
            'subset': subset,
            'metadata': sample_score.get('sample_metadata')
        },
        'metrics': score.get('value') or {},
        'model_response': {
            'text': raw_completion,
            'stop_reason': stop_reason,
            'tool_calls': message.get('tool_calls'),
        },
        'answer': {
            'outcome': outcome,
            'problem_id': str(review['index'] if group_id is None else group_id),
            'repeat_id': int(sample_score.get('generation_index') or 0),
            'ground_truth': _text(review.get('target')),
            'extracted_answer': _text(score.get('extracted_prediction')),
            'assembled_prompt': _text(messages),
            'raw_completion': raw_completion,
            'fail_reason': fail_reason,
            'generated_tokens': int((output.get('usage') or {}).get('output_tokens') or 0),
            'latency_ms': latency_ms,
        },
    }, stop_reason in {'max_tokens', 'model_length'}


def _samples(root: Path, model_id: str, benchmark: str, metric_name: str) -> tuple[list[dict[str, Any]], int, int]:
    prediction_dir = root / 'predictions' / model_id
    review_dir = root / 'reviews' / model_id
    prefix = f'{benchmark}_'
    review_paths = sorted(path for path in review_dir.glob('*.jsonl') if path.name.startswith(prefix))
    samples: list[dict[str, Any]] = []
    outcome_counts: dict[str, int] = {}
    truncated = 0
    total = 0
    for review_path in review_paths:
        prediction_path = prediction_dir / review_path.name
        # Concurrent writers may persist predictions and reviews in different orders.
        offsets = {}
        with prediction_path.open('rb') as prediction_rows:
            while line := prediction_rows.readline():
                if line.strip():
                    row = json.loads(line)
                    offsets[int(row['index'])] = prediction_rows.tell() - len(line)
            subset = review_path.stem[len(prefix):]
            with review_path.open('rb') as review_rows:
                for line in review_rows:
                    if not line.strip():
                        continue
                    review = json.loads(line)
                    prediction_rows.seek(offsets[int(review['index'])])
                    prediction = json.loads(prediction_rows.readline())
                    sample, is_truncated = _sample(len(samples), subset, review, prediction, metric_name)
                    outcome = sample['answer']['outcome']
                    total += 1
                    truncated += is_truncated
                    # Keep deterministic evidence without uploading the full benchmark.
                    if outcome_counts.get(outcome, 0) >= MAX_SAMPLES_PER_OUTCOME:
                        continue
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                    samples.append(sample)
    return samples, truncated, total


def _task_identity(benchmark, spec, identity, metadata) -> dict[str, Any]:
    weight_sha256 = metadata['weight_sha256']
    wkv_mode = metadata['wkv_mode']
    split = spec.get('eval_split')
    return {
        'identity': f'{weight_sha256}:{wkv_mode}:{benchmark}',
        'weight_sha256': weight_sha256,
        'weight_display_name': metadata['weight_display_name'],
        'wkv_mode': wkv_mode,
        'benchmark': benchmark,
        'task_name': benchmark,
        'task_version': str(identity['evaluation_version']),
        'dataset': str(spec.get('dataset_id') or benchmark),
        'subset': None,
        'evaluation_splits': [str(split)] if split else [],
        'languages': sorted(metadata.get('languages', [])),
        'tags': sorted(metadata.get('tags', [])),
    }


def publish_benchmark_callback(
    work_dir: str,
    model_id: str,
    benchmark: str,
    metadata_path: str,
) -> dict[str, Any]:
    """Read one completed benchmark's artifacts and publish them atomically."""
    root = Path(work_dir)
    config = yaml.safe_load((root / 'configs' / 'task_config.yaml').read_text(encoding='utf-8'))
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    report_path = root / 'reports' / model_id / f'{benchmark}.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    metrics, primary_metric, sample_metric, multiplier = _report_metrics(report)
    samples, truncated, total_samples = _samples(root, model_id, benchmark, sample_metric)
    spec = config['resolved_benchmarks'][benchmark]
    evaluation_identity = config['evaluation_identity']['benchmarks'][benchmark]
    task = _task_identity(benchmark, spec, evaluation_identity, metadata)
    task_config = {
        'benchmark': spec,
        'limit': config.get('limit'),
        'repeats': config.get('repeats'),
        'seed': config.get('seed')
    }
    category_id = f"benchmark_{re.sub(r'[^a-z0-9_]+', '_', benchmark.lower()).strip('_')}"
    comparison = {
        'model': metadata['comparison']['model'],
        'benchmark': {
            'label': str(report.get('dataset_pretty_name') or benchmark),
            'categories': [{
                'id': category_id,
                'label': str(report.get('dataset_pretty_name') or benchmark)
            }],
            'evaluation_method': 'cot' if (spec.get('extra_params') or {}).get('use_cot') else 'direct',
            'score_multiplier': multiplier,
        },
        'evaluation': {
            'prompt_profile': benchmark,
            'prompt_template': str(spec.get('prompt_template') or spec.get('query_template') or ''),
            'precision': metadata['wkv_mode'],
        },
        'coordinates': metadata['comparison']['coordinates'],
        'samples': len(samples),
        'truncation_rate': truncated / total_samples if total_samples else 0.0,
    }
    publication = {
        'schema_version': SCHEMA_VERSION,
        'task': task,
        'result_files': [],
        'task_config': task_config,
        'environment': {
            'framework': 'evalscope',
            'version': config.get('evalscope_version'),
            'eval_type': config.get('eval_type')
        },
        'sampling_config': config.get('generation_config') or {},
        'primary_metric': primary_metric,
        'metrics': metrics,
        'diagnostics': {
            'execution_summary': report.get('execution_summary'),
            'samples_total': total_samples,
            'samples_uploaded': len(samples)
        },
        'samples': samples,
        'comparison': comparison,
    }
    campaign = {
        'schema_version': SCHEMA_VERSION,
        'run_key': '',
        'source': 'evalscope',
        'config_sha256': _sha256(task_config),
        'registry_sha256': _sha256(evaluation_identity),
        'contract_sha256': _sha256(CONTRACT),
        'configured_benchmarks': [benchmark],
        'resolved_benchmarks': [benchmark],
        'skipped_benchmarks': [],
        'expected_tasks': [task],
        'rerun_reason': metadata.get('rerun_reason'),
    }
    run_key_payload = dict(campaign)
    run_key_payload.pop('run_key')
    campaign['run_key'] = _sha256(run_key_payload)
    base_url = os.environ['SCOREBOARD_API_BASE_URL']
    token = os.environ['SCOREBOARD_PUBLICATION_TOKEN']
    campaign_status, campaign_receipt = _send(
        base_url, token, 'POST', '/api/v1/evaluation-campaigns', f"campaign:{campaign['run_key']}", campaign
    )
    # Scoreboard requires create, one atomic task PUT, then finalize.
    campaign_id = campaign_receipt['campaign_id']
    if campaign_receipt['status'] == 'complete':
        return {'campaign': {'http_status': campaign_status, **campaign_receipt}, 'task': None, 'finalize': None}
    task_publication = {**publication, 'campaign_id': campaign_id}
    task_status, task_receipt = _send(
        base_url,
        token,
        'PUT',
        f"/api/v1/evaluation-campaigns/{campaign_id}/tasks/{quote(task['identity'], safe='')}",
        f'publish:{_sha256(task_publication)}',
        task_publication,
    )
    finalize_status, finalize_receipt = _send(
        base_url,
        token,
        'POST',
        f'/api/v1/evaluation-campaigns/{campaign_id}/finalize',
        f'finalize:{campaign_id}',
        None,
    )
    campaign_receipt['http_status'] = campaign_status
    task_receipt['http_status'] = task_status
    finalize_receipt['http_status'] = finalize_status
    return {'campaign': campaign_receipt, 'task': task_receipt, 'finalize': finalize_receipt}
