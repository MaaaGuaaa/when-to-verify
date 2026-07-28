from __future__ import annotations

from pathlib import Path

import pytest

from src.generation.sop05r_output_loader import load_sop05r_events
from src.generation.sop05r_run import publish_sop05r_generation
from tests.test_sop05r_run import _complete_publication_inputs


def test_strict_loader_reconstructs_events_and_authenticated_trajectories(
    tmp_path: Path,
) -> None:
    context, collection = _complete_publication_inputs()
    root = tmp_path / "complete"
    published = publish_sop05r_generation(root, context, collection)

    loaded = load_sop05r_events(
        root,
        require_complete=True,
        expected_publication_semantic_digest=(
            published.publication_semantic_digest
        ),
        expected_run_id=context.run_id,
    )

    assert loaded.complete
    assert tuple(event.generated_event_id for event in loaded.events) == (
        collection.selected_events[0].event.generated_event_id,
    )
    assert tuple(record.event_id for record in loaded.trajectory_store.records) == (
        loaded.events[0].generated_event_id,
    )
    assert loaded.events[0].world.world_id in loaded.target_motion.worlds
    assert loaded.summary["selected_count"] == 1


def test_loader_detects_outer_checksum_tampering(tmp_path: Path) -> None:
    context, collection = _complete_publication_inputs()
    root = tmp_path / "tampered"
    publish_sop05r_generation(root, context, collection)
    with (root / "events.jsonl").open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(ValueError, match="checksum"):
        load_sop05r_events(root, require_complete=True)


def test_partial_publication_requires_explicit_incomplete_loading(
    tmp_path: Path,
) -> None:
    context, collection = _complete_publication_inputs(accepted_quota=2)
    root = tmp_path / "partial"
    publish_sop05r_generation(root, context, collection)

    with pytest.raises(ValueError, match="completion"):
        load_sop05r_events(root, require_complete=True)
    loaded = load_sop05r_events(root, require_complete=False)
    assert not loaded.complete
    assert loaded.summary["quota_met"] is False


def test_loader_rejects_wrong_expected_identity(tmp_path: Path) -> None:
    context, collection = _complete_publication_inputs()
    root = tmp_path / "identity"
    publish_sop05r_generation(root, context, collection)

    with pytest.raises(ValueError, match="run_id"):
        load_sop05r_events(
            root,
            require_complete=True,
            expected_run_id="sop05r-run-other",
        )
    with pytest.raises(ValueError, match="publication semantic digest"):
        load_sop05r_events(
            root,
            require_complete=True,
            expected_publication_semantic_digest="f" * 64,
        )
