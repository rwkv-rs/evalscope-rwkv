from __future__ import annotations

import gzip
import hashlib
import json
from typing import Any

import pytest

from scripts.upload_score import canonical_json, publish_bundle

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
