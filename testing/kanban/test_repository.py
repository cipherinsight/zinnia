from __future__ import annotations

from pathlib import Path

from kanban.repository import KanbanRepo, discover_repo_root


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repo_board_validates_cleanly():
    repo = KanbanRepo(REPO_ROOT)
    assert repo.validate() == []


def test_discover_repo_root_finds_board_from_nested_path():
    nested = REPO_ROOT / "testing" / "semantics" / "array"
    assert discover_repo_root(nested) == REPO_ROOT


def test_find_card_by_id_and_slug():
    repo = KanbanRepo(REPO_ROOT)
    by_id = repo.find_card("array.nested-view-writeback")
    by_slug = repo.find_card("nested-view-writeback")
    assert by_id.id == by_slug.id
    assert by_id.category == "array-semantics"

