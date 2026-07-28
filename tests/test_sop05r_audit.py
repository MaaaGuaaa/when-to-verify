from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.sop05r_audit import (
    build_sop05r_audit_metrics,
    load_sop05r_audit,
    publish_sop05r_audit,
)
from src.generation.sop05r_output_loader import load_sop05r_events
from src.generation.sop05r_run import publish_sop05r_generation
from tests.test_sop05r_run import _complete_publication_inputs
from tests.test_sop05r_visuals import _visual_request


@pytest.fixture(scope="module")
def loaded_source(tmp_path_factory: pytest.TempPathFactory):
    context, collection = _complete_publication_inputs()
    root = tmp_path_factory.mktemp("sop05r-audit-source") / "source"
    publish_sop05r_generation(root, context, collection)
    return load_sop05r_events(root, require_complete=True)


def test_audit_metrics_recompute_all_denominators_and_sample_identity(
    loaded_source,
) -> None:
    metrics = build_sop05r_audit_metrics(loaded_source)

    assert metrics.sample_ids == tuple(
        event.generated_event_id for event in loaded_source.events
    )
    assert len(metrics.sample_id_digest) == 64
    assert metrics.counts["input_base_count"] == 1
    assert metrics.counts["template_count"] == 1
    assert metrics.counts["selected_count"] == 1
    assert metrics.rates["planner_success_given_geometry"] == 1.0
    assert metrics.rates["end_to_end_acceptance"] == 1.0
    assert metrics.history_counts == {
        "seen_then_occluded": 1,
        "unseen_in_history_window": 0,
    }
    assert metrics.active_revealable_fraction == 1.0
    assert metrics.attempts["median"] == 1.0


def test_audit_publication_binds_source_ids_checksums_and_completion(
    tmp_path: Path,
    loaded_source,
) -> None:
    output = tmp_path / "audit"
    request = _visual_request()
    request = request.with_event(loaded_source.events[0])

    result = publish_sop05r_audit(
        output,
        source=loaded_source,
        visual_requests=(request,),
        required_sample_count=1,
    )
    loaded = load_sop05r_audit(output, require_complete=True)

    assert result.status == "complete"
    assert result.exit_code == 0
    assert loaded["selected_event_ids"] == [
        loaded_source.events[0].generated_event_id
    ]
    assert (output / ".sop05r-audit-complete").is_file()
    assert (output / "artifact_checksums.sha256").is_file()
    sample = loaded["samples"][0]
    assert sample["event_replay"]["sha256"]
    assert sample["paired_events"]["sha256"]
    with pytest.raises(FileExistsError):
        publish_sop05r_audit(
            output,
            source=loaded_source,
            visual_requests=(request,),
            required_sample_count=1,
        )


def test_audit_loader_detects_visual_tampering(
    tmp_path: Path,
    loaded_source,
) -> None:
    output = tmp_path / "tampered"
    request = _visual_request().with_event(loaded_source.events[0])
    publish_sop05r_audit(
        output,
        source=loaded_source,
        visual_requests=(request,),
        required_sample_count=1,
    )
    replay = next(output.glob("samples/*/event_replay.gif"))
    payload = bytearray(replay.read_bytes())
    payload[-1] ^= 1
    replay.write_bytes(payload)

    with pytest.raises(ValueError, match="checksum"):
        load_sop05r_audit(output, require_complete=True)
