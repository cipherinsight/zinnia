from __future__ import annotations

from datetime import date
from pathlib import Path

from kanban.cli import main
from kanban.repository import KanbanRepo


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_validate_on_repo_board(capsys):
    exit_code = main(["--root", str(REPO_ROOT), "validate"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "validated" in captured.out


def test_cli_show_outputs_card_details(capsys):
    exit_code = main(
        ["--root", str(REPO_ROOT), "show", "array.nested-view-writeback"]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Nested view writeback / true aliasing" in captured.out
    assert "status: deferred" in captured.out


def test_cli_move_updates_card_status(tmp_path, capsys):
    root = _create_temp_board(tmp_path)
    exit_code = main(
        [
            "--root",
            str(root),
            "move",
            "tooling.temp-card",
            "verify",
            "--updated",
            "2026-05-06",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "moved tooling.temp-card -> verify" in captured.out

    repo = KanbanRepo(root)
    card = repo.find_card("tooling.temp-card")
    assert card.status == "verify"
    assert card.updated == date(2026, 5, 6)


def _create_temp_board(root: Path) -> Path:
    board_dir = root / "kanban"
    card_dir = board_dir / "cards" / "tooling" / "temp-card"
    card_dir.mkdir(parents=True)

    (board_dir / "config.toml").write_text(
        "\n".join(
            [
                'version = 1',
                'name = "Temp Board"',
                'cards_dir = "kanban/cards"',
                'metadata_file = "card.toml"',
                'description_file = "README.md"',
                'default_status = "intake"',
                'done_statuses = ["done"]',
                'statuses = ["intake", "triaged", "ready", "in_progress", "blocked", "verify", "done", "deferred"]',
                'categories = ["tooling"]',
                'priorities = ["p0", "p1", "p2", "p3"]',
                'sizes = ["small", "medium", "large", "hard"]',
                'types = ["bug", "feature", "task", "design", "research"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    (card_dir / "card.toml").write_text(
        "\n".join(
            [
                'id = "tooling.temp-card"',
                'title = "Temp card"',
                'status = "intake"',
                'category = "tooling"',
                'type = "task"',
                'priority = "p2"',
                'size = "small"',
                'created = "2026-05-05"',
                'updated = "2026-05-05"',
                'owner = ""',
                'summary = "Temporary card for move command tests."',
                'labels = ["temp"]',
                'depends_on = []',
                'blocked_by = []',
                'related_tests = []',
                'related_files = []',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (card_dir / "README.md").write_text("## Problem\nTemp\n", encoding="utf-8")
    return root
