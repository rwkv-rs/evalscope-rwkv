"""Publish a Scoreboard campaign and its task publications over HTTP."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')


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
    headers = {
        'Authorization': f'Bearer {token}',
        'Idempotency-Key': idempotency_key,
    }
    if body is not None:
        headers.update({
            'Content-Type': 'application/json',
            'Content-Encoding': 'gzip',
        })
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
    tasks: Sequence[Mapping[str, Any]],
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('tasks', type=Path, nargs='+')
    args = parser.parse_args(argv)
    receipt = publish_bundle(
        load_json(args.campaign),
        [load_json(path) for path in args.tasks],
        os.environ['SCOREBOARD_API_BASE_URL'],
        os.environ['SCOREBOARD_PUBLICATION_TOKEN'],
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
