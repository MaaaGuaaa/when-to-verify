"""Immutable cross-lane SOP06 history catalog tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import src.datasets.sop06_history_catalog as catalog_module
import src.generation.sop06_history_release as release_module
from src.datasets.sop06_history_catalog import (
    load_sop06_history_catalog,
    publish_sop06_history_catalog,
)
from src.generation.sop06_history_release import (
    publish_sop06_history_release,
)
from tests.test_sop06_history_release import (
    _accepted,
    _render_one,
    _request,
    _source,
)


def test_catalog_references_natural_and_supplement_without_copying(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(release_module, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(catalog_module, "_REPOSITORY_ROOT", tmp_path)
    current = _source(2)
    monkeypatch.setattr(
        release_module,
        "load_sop06_finalized_source",
        lambda **kwargs: current,
    )
    natural_request = replace(
        _request(tmp_path),
        source_root=Path("outputs/natural-mother"),
        final_scenario_root=Path("outputs/natural-final"),
        output_dir=tmp_path / "natural",
    )
    publish_sop06_history_release(natural_request, render_one=_render_one)

    supplement_accepted = tuple(
        replace(
            _accepted(index),
            mother_id=f"supplement-mother-{index}",
            scenario_id=f"supplement-scenario-{index}",
        )
        for index in range(3)
    )
    current = SimpleNamespace(
        source_mode="complete_mother",
        source_publication_semantic_digest="e" * 64,
        final_release_identity="f" * 64,
        base_config={},
        accepted=supplement_accepted,
    )
    supplement_request = replace(
        natural_request,
        source_family="a_supplement",
        source_root=Path("outputs/supplement-mother"),
        final_scenario_root=Path("outputs/supplement-final"),
        output_dir=tmp_path / "supplement",
    )
    publish_sop06_history_release(supplement_request, render_one=_render_one)

    monkeypatch.delattr(
        catalog_module,
        "load_sop06_history_release",
        raising=False,
    )
    monkeypatch.delattr(
        catalog_module,
        "load_sop06_history_shard",
        raising=False,
    )

    output = tmp_path / "catalog"
    publish_sop06_history_catalog(
        (natural_request.output_dir, supplement_request.output_dir),
        output,
    )
    loaded = load_sop06_history_catalog(output)

    assert [entry.source_family for entry in loaded.entries] == [
        "a_supplement",
        "natural",
    ]
    assert loaded.sample_count == 5
    assert loaded.entry_count == 2
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "checksums.json",
        "COMPLETE.json",
    }
