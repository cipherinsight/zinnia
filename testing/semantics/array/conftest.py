from __future__ import annotations

from pathlib import Path

from .reporting import write_report


def pytest_sessionfinish(session, exitstatus):
    cache_dir = Path(session.config.getini("cache_dir"))
    if not cache_dir.is_absolute():
        cache_dir = Path(session.config.rootpath) / cache_dir
    report_path = cache_dir / "array_semantics_report.json"
    write_report(report_path)
