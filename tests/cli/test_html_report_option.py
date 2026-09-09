from __future__ import annotations

import sys

from evalscope.arguments import parse_args
from evalscope.config import TaskConfig


def test_html_report_enabled_by_default(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['evalscope'])

    args = parse_args()

    assert args.html_report is True
    assert TaskConfig().html_report is True


def test_no_html_report_disables_standalone_report(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['evalscope', '--no-html-report'])

    args = parse_args()

    assert args.html_report is False
    assert TaskConfig(html_report=args.html_report).html_report is False
