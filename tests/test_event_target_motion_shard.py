from __future__ import annotations

import importlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from src.contracts import (
    GridSpec,
    OracleContext,
    OracleWorld,
    load_dataclass,
    save_dataclass,
)


def _sut():
    return importlib.import_module("src.generation.event_target_motion_shard")


def _grid() -> GridSpec:
    return GridSpec(
        height=8,
        width=8,
        history_steps=8,
        future_steps=15,
        resolution_m=0.1,
    )


def _motion(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = np.column_stack(
        (
            np.arange(8, dtype=np.float32) * np.float32(0.2) + offset,
            np.arange(8, dtype=np.float32) * np.float32(0.05),
            np.arange(8, dtype=np.float32) * np.float32(0.01),
        )
    ).astype(np.float32)
    current = history[7].copy()
    future = np.column_stack(
        (
            current[0] + np.arange(1, 16, dtype=np.float32) * np.float32(0.2),
            current[1] + np.arange(1, 16, dtype=np.float32) * np.float32(0.05),
            current[2] + np.arange(1, 16, dtype=np.float32) * np.float32(0.01),
        )
    ).astype(np.float32)
    return history, current, future


def _record(
    module,
    suffix: str = "a",
    *,
    target_id: str | None = None,
    policy_digest: str | None = None,
    history: np.ndarray | None = None,
    current: np.ndarray | None = None,
    future: np.ndarray | None = None,
):
    default_history, default_current, default_future = _motion(
        float(ord(suffix[0]) - ord("a"))
    )
    spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.3},
    }
    return module.create_event_target_motion_record(
        generated_event_id=f"event-{suffix}",
        world_id=f"world-{suffix}",
        base_state_id=f"base-{suffix}",
        trajectory_id=f"trajectory-{suffix}",
        target_dynamic_object_id=target_id or f"target-{suffix}",
        source_snippet_id=f"snippet-{suffix}",
        source_object_id=f"recording-{suffix}::object",
        object_type="human",
        footprint_spec=spec,
        footprint_spec_digest=module.compute_footprint_spec_digest(spec),
        target_type_policy_digest=(
            f"{ord(suffix[0]):032x}" if policy_digest is None else policy_digest
        ),
        history_poses=default_history if history is None else history,
        current_pose=default_current if current is None else current,
        future_poses=default_future if future is None else future,
    )


def _world(record, grid: GridSpec, *, context: bool = True) -> OracleWorld:
    trajectories = {record.target_dynamic_object_id: record.future_poses.copy()}
    specs = {record.target_dynamic_object_id: dict(record.footprint_spec)}
    if context:
        trajectories["context-object"] = np.zeros(
            (grid.future_steps, 3), dtype=np.float32
        )
        specs["context-object"] = {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.25},
        }
    metadata = {
        "generated_event_id": record.generated_event_id,
        "world_id": record.world_id,
        "base_state_id": record.base_state_id,
        "trajectory_id": record.trajectory_id,
        "target_dynamic_object_id": record.target_dynamic_object_id,
        "source_snippet_id": record.source_snippet_id,
        "source_object_id": record.source_object_id,
        "target_object_type": record.object_type,
        "target_footprint_spec": dict(record.footprint_spec),
        "target_footprint_spec_digest": record.footprint_spec_digest,
        "target_type_policy_digest": record.target_type_policy_digest,
        "event_target_motion_layout_version": record.layout_version,
        "target_history_array_digest": record.history_array_digest,
        "target_future_array_digest": record.future_array_digest,
        "target_motion_record_digest": record.record_digest,
        "target_current_pose": [float(value) for value in record.current_pose],
    }
    return OracleWorld(
        world_id=record.world_id,
        base_state_id=record.base_state_id,
        static_occupancy=np.zeros((grid.height, grid.width), dtype=np.float32),
        dynamic_object_trajectories=trajectories,
        dynamic_object_specs=specs,
        occluders=(),
        blind_spot_config={"kind": "structural"},
        random_seed=17,
        metadata=metadata,
    )


def _write(module, root: Path, records, worlds):
    return module.write_event_target_motion_shard(
        records, worlds, root, grid=_grid()
    )


def _rewrite_npz(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as payload:
        copied = {name: payload[name].copy() for name in payload.files}
    mutate(copied)
    with path.open("wb") as handle:
        np.savez(handle, **copied)


def _rewrite_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rewrite_manifest(path: Path, mutate) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_factory_freezes_contract_and_owns_c_contiguous_float32_copies() -> None:
    module = _sut()
    base_history, base_current, base_future = _motion()
    history = np.asfortranarray(base_history)
    current = base_current.copy()
    future = np.asfortranarray(base_future)
    record = _record(
        module, history=history, current=current, future=future
    )

    history[:] = -99.0
    current[:] = -99.0
    future[:] = -99.0

    assert record.schema_version == "2.0.0"
    assert record.layout_version == "event_target_motion_history8_future15_v1"
    assert record.history_poses.shape == (8, 3)
    assert record.current_pose.shape == (3,)
    assert record.future_poses.shape == (15, 3)
    for array in (record.history_poses, record.current_pose, record.future_poses):
        assert array.dtype == np.float32
        assert array.flags.c_contiguous
        assert array.flags.owndata
        assert not array.flags.writeable
        assert np.isfinite(array).all()
    np.testing.assert_array_equal(record.current_pose, record.history_poses[7])
    assert not np.all(record.history_poses == -99.0)
    with pytest.raises(FrozenInstanceError):
        record.world_id = "other-world"
    with pytest.raises(TypeError):
        record.footprint_spec["object_type"] = "carried_object"
    with pytest.raises(TypeError):
        record.footprint_spec["footprint"]["radius_m"] = 9.0


def test_world_metadata_builder_freezes_v1_join_fields_without_mutable_aliases() -> None:
    module = _sut()
    record = _record(module)

    metadata = module.build_event_target_motion_world_metadata(record)

    assert set(metadata) == {
        "generated_event_id",
        "world_id",
        "base_state_id",
        "trajectory_id",
        "target_dynamic_object_id",
        "source_snippet_id",
        "source_object_id",
        "target_object_type",
        "target_footprint_spec",
        "target_footprint_spec_digest",
        "target_type_policy_digest",
        "event_target_motion_layout_version",
        "target_history_array_digest",
        "target_future_array_digest",
        "target_motion_record_digest",
        "target_current_pose",
    }
    assert metadata["generated_event_id"] == record.generated_event_id
    assert metadata["world_id"] == record.world_id
    assert metadata["target_footprint_spec"] == record.footprint_spec
    assert metadata["target_footprint_spec"] is not record.footprint_spec
    assert metadata["target_footprint_spec"]["footprint"] is not (
        record.footprint_spec["footprint"]
    )
    assert metadata["target_current_pose"] == [
        float(value) for value in record.current_pose
    ]
    assert all(type(value) is float for value in metadata["target_current_pose"])

    metadata["target_footprint_spec"]["footprint"]["radius_m"] = 9.0
    metadata["target_current_pose"][0] = 9.0
    assert record.footprint_spec["footprint"]["radius_m"] == pytest.approx(0.3)
    assert float(record.current_pose[0]) != 9.0


@pytest.mark.parametrize(
    ("metadata_key", "bad_value"),
    [
        ("generated_event_id", "wrong-event"),
        ("world_id", "wrong-world"),
        ("base_state_id", "wrong-base"),
        ("trajectory_id", "wrong-trajectory"),
        ("target_dynamic_object_id", "wrong-target"),
        ("source_snippet_id", "wrong-snippet"),
        ("source_object_id", "wrong-source-object"),
        ("target_object_type", "carried_object"),
        (
            "target_footprint_spec",
            {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.31},
            },
        ),
        ("target_footprint_spec_digest", "0" * 32),
        ("target_type_policy_digest", "0" * 32),
        ("event_target_motion_layout_version", "wrong-layout"),
        ("target_history_array_digest", "0" * 32),
        ("target_future_array_digest", "0" * 32),
        ("target_motion_record_digest", "0" * 32),
        ("target_current_pose", [9.0, 9.0, 9.0]),
    ],
)
def test_public_record_world_join_accepts_canonical_pair_and_rejects_identity_tampering(
    metadata_key: str, bad_value: object
) -> None:
    module = _sut()
    record = _record(module)
    world = _world(record, _grid())

    module.validate_event_target_motion_world_join(record, world, _grid())

    metadata = dict(world.metadata)
    metadata[metadata_key] = bad_value
    tampered = replace(world, metadata=metadata)
    with pytest.raises(ValueError):
        module.validate_event_target_motion_world_join(record, tampered, _grid())


def test_public_record_world_join_rejects_current_metadata_that_only_matches_after_float32_rounding() -> None:
    module = _sut()
    record = _record(module)
    world = _world(record, _grid())
    metadata = dict(world.metadata)
    tampered_current = list(metadata["target_current_pose"])
    canonical_value = tampered_current[0]
    tampered_current[0] = canonical_value + 1e-12
    assert tampered_current[0] != canonical_value
    assert np.float32(tampered_current[0]) == record.current_pose[0]
    metadata["target_current_pose"] = tampered_current

    with pytest.raises(
        ValueError, match="target_current_pose metadata mismatch"
    ):
        module.validate_event_target_motion_world_join(
            record, replace(world, metadata=metadata), _grid()
        )


@pytest.mark.parametrize(
    "field_name",
    ["history_poses", "current_pose", "future_poses"],
)
def test_validator_and_writer_reject_writeable_record_arrays(
    tmp_path: Path, field_name: str
) -> None:
    module = _sut()
    record = _record(module)
    writeable_copy = getattr(record, field_name).copy()
    assert writeable_copy.flags.writeable
    invalid = replace(record, **{field_name: writeable_copy})

    with pytest.raises(ValueError, match=rf"{field_name}.*writeable=False"):
        module.validate_event_target_motion_record(invalid)
    with pytest.raises(ValueError, match=rf"{field_name}.*writeable=False"):
        _write(
            module,
            tmp_path / field_name,
            [invalid],
            [_world(invalid, _grid())],
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("history", np.zeros((7, 3), dtype=np.float32), "history_poses shape"),
        ("current", np.zeros((4,), dtype=np.float32), "current_pose shape"),
        ("future", np.zeros((14, 3), dtype=np.float32), "future_poses shape"),
        ("history", np.zeros((8, 3), dtype=np.float64), "history_poses dtype"),
        ("current", np.zeros((3,), dtype=np.float64), "current_pose dtype"),
        ("future", np.zeros((15, 3), dtype=np.float64), "future_poses dtype"),
    ],
)
def test_factory_rejects_wrong_shape_or_dtype_without_casting(
    field: str, bad_value: np.ndarray, message: str
) -> None:
    module = _sut()
    history, current, future = _motion()
    values = {"history": history, "current": current, "future": future}
    values[field] = bad_value
    with pytest.raises(ValueError, match=message):
        _record(
            module,
            history=values["history"],
            current=values["current"],
            future=values["future"],
        )


@pytest.mark.parametrize("kind", ["history", "current", "future", "seam"])
def test_factory_rejects_nonfinite_arrays_and_current_seam(kind: str) -> None:
    module = _sut()
    history, current, future = _motion()
    if kind == "history":
        history[0, 0] = np.nan
    elif kind == "current":
        current[0] = np.inf
    elif kind == "future":
        future[-1, 2] = -np.inf
    else:
        current[0] += np.float32(0.01)
    message = "finite" if kind != "seam" else r"history_poses\[7\]"
    with pytest.raises(ValueError, match=message):
        _record(module, history=history, current=current, future=future)


@pytest.mark.parametrize(
    "bad_digest",
    ["short", "g" * 32, "A" * 32],
    ids=["short", "non-hex", "uppercase"],
)
def test_policy_digest_must_be_exact_lowercase_blake2b128_hex(
    bad_digest: str,
) -> None:
    module = _sut()
    with pytest.raises(ValueError, match="target_type_policy_digest"):
        _record(module, policy_digest=bad_digest)

    valid = _record(module)
    invalid = replace(valid, target_type_policy_digest=bad_digest)
    with pytest.raises(ValueError, match="target_type_policy_digest"):
        module.validate_event_target_motion_record(invalid)


def test_array_and_record_digests_are_domain_separated_and_semantic() -> None:
    module = _sut()
    history, _, _ = _motion()
    same_values_fortran = np.asfortranarray(history)
    history_digest = module.compute_motion_array_digest(
        history, field_name="target_history_poses"
    )
    assert history_digest == module.compute_motion_array_digest(
        same_values_fortran, field_name="target_history_poses"
    )
    assert history_digest != module.compute_motion_array_digest(
        history, field_name="target_future_poses"
    )
    changed = history.copy()
    changed[0, 0] += np.float32(0.01)
    assert history_digest != module.compute_motion_array_digest(
        changed, field_name="target_history_poses"
    )
    first = _record(module, "a")
    second = _record(module, "b", target_id=first.target_dynamic_object_id)
    assert first.record_digest != second.record_digest


def test_oracle_world_digest_is_canonical_and_binds_array_representation() -> None:
    module = _sut()
    world = _world(_record(module), _grid())
    reordered = replace(
        world,
        dynamic_object_trajectories=dict(
            reversed(tuple(world.dynamic_object_trajectories.items()))
        ),
        dynamic_object_specs=dict(
            reversed(tuple(world.dynamic_object_specs.items()))
        ),
        metadata=dict(reversed(tuple(world.metadata.items()))),
    )
    digest = module.compute_oracle_world_semantic_digest(world)
    assert digest == module.compute_oracle_world_semantic_digest(reordered)

    changed_values = world.static_occupancy.copy()
    changed_values[0, 0] = np.float32(1.0)
    assert digest != module.compute_oracle_world_semantic_digest(
        replace(world, static_occupancy=changed_values)
    )
    assert digest != module.compute_oracle_world_semantic_digest(
        replace(world, static_occupancy=world.static_occupancy.astype(np.float64))
    )
    assert digest != module.compute_oracle_world_semantic_digest(
        replace(world, static_occupancy=world.static_occupancy.reshape(4, 16))
    )
    assert digest != module.compute_oracle_world_semantic_digest(
        replace(world, static_occupancy=None)
    )


def test_writer_and_loader_round_trip_exact_single_version_contract(tmp_path: Path) -> None:
    module = _sut()
    record = _record(module)
    world = _world(record, _grid())
    output = tmp_path / "shard"
    paths = _write(module, output, [record], [world])

    assert set(paths) == {"directory", "manifest", "payload", "summary", "worlds"}
    assert sorted(path.name for path in output.iterdir()) == [
        "event_target_motion_history8_future15_v1.npz",
        "generated_event_manifest.jsonl",
        "oracle_worlds",
        "shard_summary.json",
    ]
    with np.load(paths["payload"], allow_pickle=False) as payload:
        assert set(payload.files) == {
            "history_poses",
            "current_poses",
            "future_poses",
            "meta_json",
        }
        assert payload["history_poses"].shape == (1, 8, 3)
        assert payload["current_poses"].shape == (1, 3)
        assert payload["future_poses"].shape == (1, 15, 3)
        assert payload["history_poses"].dtype == np.float32
        assert payload["current_poses"].dtype == np.float32
        assert payload["future_poses"].dtype == np.float32
        meta = json.loads(str(payload["meta_json"]))
    assert meta["history_time_offsets_s"] == pytest.approx(
        [-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0]
    )
    assert meta["future_time_offsets_s"] == pytest.approx(
        [0.2 * index for index in range(1, 16)]
    )
    manifest_row = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest_row["world_semantic_digest"] == (
        module.compute_oracle_world_semantic_digest(world)
    )

    loaded = module.load_event_target_motion_shard(
        output,
        grid=_grid(),
        expected_base_state_ids={record.base_state_id},
        expected_trajectory_ids={record.trajectory_id},
    )
    assert loaded.records == (loaded.records[0],)
    assert loaded.records[0].generated_event_id == record.generated_event_id
    assert loaded.records[0].record_digest == record.record_digest
    assert manifest_row["footprint_spec"] == record.footprint_spec
    np.testing.assert_array_equal(loaded.records[0].history_poses, record.history_poses)
    np.testing.assert_array_equal(loaded.records[0].future_poses, record.future_poses)
    for array in (
        loaded.records[0].history_poses,
        loaded.records[0].current_pose,
        loaded.records[0].future_poses,
    ):
        assert not array.flags.writeable
    with pytest.raises(TypeError):
        loaded.records[0].footprint_spec["footprint"]["radius_m"] = 9.0
    assert tuple(loaded.worlds) == (record.world_id,)


def test_writer_rejects_empty_duplicate_event_or_world_but_allows_target_reuse(
    tmp_path: Path,
) -> None:
    module = _sut()
    with pytest.raises(ValueError, match="records must not be empty"):
        _write(module, tmp_path / "empty", [], [])

    first = _record(module, "a", target_id="shared-target")
    second = _record(module, "b", target_id="shared-target")
    _write(
        module,
        tmp_path / "shared-target-ok",
        [first, second],
        [_world(first, _grid()), _world(second, _grid())],
    )

    duplicate_event = replace(second, generated_event_id=first.generated_event_id)
    with pytest.raises(ValueError, match="duplicate generated_event_id"):
        _write(
            module,
            tmp_path / "duplicate-event",
            [first, duplicate_event],
            [_world(first, _grid()), _world(duplicate_event, _grid())],
        )
    duplicate_world = replace(second, world_id=first.world_id)
    with pytest.raises(ValueError, match="duplicate world_id"):
        _write(
            module,
            tmp_path / "duplicate-world",
            [first, duplicate_world],
            [_world(first, _grid()), _world(duplicate_world, _grid())],
        )


def test_writer_prevalidates_every_join_and_refuses_overwrite(tmp_path: Path) -> None:
    module = _sut()
    first = _record(module, "a")
    second = _record(module, "b")
    bad_world = _world(second, _grid())
    bad_world.dynamic_object_trajectories[second.target_dynamic_object_id][0, 0] += 1.0
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "invalid"
    with pytest.raises(ValueError, match="target future"):
        _write(
            module,
            output,
            [first, second],
            [_world(first, _grid()), bad_world],
        )
    assert not output.exists()
    assert list(parent.iterdir()) == []

    existing = parent / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write(module, existing, [first], [_world(first, _grid())])
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("bad_metadata", "message"),
    [
        (float("nan"), "finite"),
        ({1: "silently-stringified-key"}, "string keys"),
    ],
    ids=["nonfinite", "non-string-key"],
)
def test_writer_rejects_noncanonical_world_metadata(
    tmp_path: Path, bad_metadata: object, message: str
) -> None:
    module = _sut()
    record = _record(module)
    world = _world(record, _grid())
    metadata = dict(world.metadata)
    metadata["bad_metadata"] = bad_metadata
    invalid_world = replace(world, metadata=metadata)
    output = tmp_path / "invalid-world-metadata"

    with pytest.raises(ValueError, match=message):
        _write(module, output, [record], [invalid_world])
    assert not output.exists()


def test_canonical_outputs_are_order_independent_and_use_semantic_payload_digest(
    tmp_path: Path,
) -> None:
    module = _sut()
    first = _record(module, "a")
    second = _record(module, "b")
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write(
        module,
        left,
        [second, first],
        [_world(second, _grid()), _world(first, _grid())],
    )
    _write(
        module,
        right,
        [first, second],
        [_world(first, _grid()), _world(second, _grid())],
    )
    assert (left / "generated_event_manifest.jsonl").read_bytes() == (
        right / "generated_event_manifest.jsonl"
    ).read_bytes()
    assert (left / "shard_summary.json").read_bytes() == (
        right / "shard_summary.json"
    ).read_bytes()
    loaded_left = module.load_event_target_motion_shard(left, grid=_grid())
    loaded_right = module.load_event_target_motion_shard(right, grid=_grid())
    assert loaded_left.manifest_digest == loaded_right.manifest_digest
    assert loaded_left.payload_semantic_digest == loaded_right.payload_semantic_digest
    assert [record.generated_event_id for record in loaded_left.records] == [
        "event-a",
        "event-b",
    ]


def test_world_semantics_are_bound_into_manifest_identity(tmp_path: Path) -> None:
    module = _sut()
    record = _record(module)
    original_world = _world(record, _grid())
    changed_trajectories = {
        key: value.copy()
        for key, value in original_world.dynamic_object_trajectories.items()
    }
    changed_trajectories["context-object"][0, 0] += np.float32(0.25)
    changed_world = replace(
        original_world,
        dynamic_object_trajectories=changed_trajectories,
    )
    original_output = tmp_path / "original"
    changed_output = tmp_path / "changed"
    _write(module, original_output, [record], [original_world])
    _write(module, changed_output, [record], [changed_world])

    original_row = json.loads(
        (original_output / "generated_event_manifest.jsonl").read_text(
            encoding="utf-8"
        )
    )
    changed_row = json.loads(
        (changed_output / "generated_event_manifest.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert original_row["record_digest"] == changed_row["record_digest"]
    assert original_row["world_semantic_digest"] != (
        changed_row["world_semantic_digest"]
    )
    assert original_row["manifest_digest"] != changed_row["manifest_digest"]


@pytest.mark.parametrize("tamper", ["row_index", "record_digest", "extra_key"])
def test_loader_rejects_manifest_tampering(tmp_path: Path, tamper: str) -> None:
    module = _sut()
    record = _record(module)
    output = tmp_path / tamper
    _write(module, output, [record], [_world(record, _grid())])
    manifest = output / "generated_event_manifest.jsonl"

    def mutate(rows):
        if tamper == "row_index":
            rows[0]["row_index"] = 2
        elif tamper == "record_digest":
            rows[0]["record_digest"] = "0" * 32
        else:
            rows[0]["unexpected"] = True

    _rewrite_manifest(manifest, mutate)
    with pytest.raises(ValueError):
        module.load_event_target_motion_shard(output, grid=_grid())


def test_loader_rejects_raw_manifest_identity_type_before_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    record = _record(module)
    output = tmp_path / "identity-type"
    _write(module, output, [record], [_world(record, _grid())])
    manifest = output / "generated_event_manifest.jsonl"
    _rewrite_manifest(
        manifest,
        lambda rows: rows[0].update(generated_event_id=7),
    )

    def digest_must_not_run(*args, **kwargs):
        raise AssertionError("manifest digest ran before raw row validation")

    monkeypatch.setattr(module, "_manifest_digest", digest_must_not_run)
    with pytest.raises(ValueError, match="generated_event_id must be a non-empty string"):
        module.load_event_target_motion_shard(output, grid=_grid())


@pytest.mark.parametrize(
    "tamper", ["extra_key", "history_value", "history_dtype", "layout", "time_offsets"]
)
def test_loader_rejects_npz_payload_tampering(tmp_path: Path, tamper: str) -> None:
    module = _sut()
    record = _record(module)
    output = tmp_path / tamper
    _write(module, output, [record], [_world(record, _grid())])
    payload_path = output / "event_target_motion_history8_future15_v1.npz"

    def mutate(payload):
        if tamper == "extra_key":
            payload["unexpected"] = np.asarray([1], dtype=np.int32)
        elif tamper == "history_value":
            payload["history_poses"][0, 0, 0] += np.float32(0.1)
        elif tamper == "history_dtype":
            payload["history_poses"] = payload["history_poses"].astype(np.float64)
        else:
            meta = json.loads(str(payload["meta_json"]))
            if tamper == "layout":
                meta["layout_version"] = "other-layout"
            else:
                meta["future_time_offsets_s"][0] = 0.0
            payload["meta_json"] = np.asarray(
                json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )

    _rewrite_npz(payload_path, mutate)
    with pytest.raises(ValueError):
        module.load_event_target_motion_shard(output, grid=_grid())


def test_loader_rejects_summary_and_world_file_set_tampering(tmp_path: Path) -> None:
    module = _sut()
    record = _record(module)
    output = tmp_path / "summary"
    _write(module, output, [record], [_world(record, _grid())])
    _rewrite_json(output / "shard_summary.json", lambda payload: payload.update(record_count=2))
    with pytest.raises(ValueError, match="summary"):
        module.load_event_target_motion_shard(output, grid=_grid())

    output = tmp_path / "world-files"
    _write(module, output, [record], [_world(record, _grid())])
    (output / "oracle_worlds" / "extra.npz").write_bytes(b"not-an-npz")
    with pytest.raises(ValueError, match="world file set"):
        module.load_event_target_motion_shard(output, grid=_grid())


@pytest.mark.parametrize("tamper", ["metadata", "future", "spec"])
def test_loader_rejects_oracle_world_join_tampering(tmp_path: Path, tamper: str) -> None:
    module = _sut()
    record = _record(module)
    output = tmp_path / tamper
    _write(module, output, [record], [_world(record, _grid())])
    world_path = output / "oracle_worlds" / f"{record.world_id}.npz"
    world = load_dataclass(world_path)
    if tamper == "metadata":
        metadata = dict(world.metadata)
        metadata["source_snippet_id"] = "wrong-snippet"
        world = replace(world, metadata=metadata)
    elif tamper == "future":
        trajectories = {
            key: value.copy() for key, value in world.dynamic_object_trajectories.items()
        }
        trajectories[record.target_dynamic_object_id][0, 0] += np.float32(0.5)
        world = replace(world, dynamic_object_trajectories=trajectories)
    else:
        specs = {key: dict(value) for key, value in world.dynamic_object_specs.items()}
        specs[record.target_dynamic_object_id] = {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.4},
        }
        world = replace(world, dynamic_object_specs=specs)
    save_dataclass(world, world_path)
    with pytest.raises(ValueError):
        module.load_event_target_motion_shard(output, grid=_grid())


@pytest.mark.parametrize(
    "tamper",
    [
        "context_trajectory",
        "context_spec",
        "static_occupancy",
        "occluders",
        "structural_blind_spot_config",
        "random_seed",
        "metadata",
    ],
)
def test_loader_rejects_semantically_valid_oracle_world_digest_tampering(
    tmp_path: Path, tamper: str
) -> None:
    module = _sut()
    record = _record(module)
    output = tmp_path / tamper
    _write(module, output, [record], [_world(record, _grid())])
    world_path = output / "oracle_worlds" / f"{record.world_id}.npz"
    world = load_dataclass(world_path)
    if tamper == "context_trajectory":
        trajectories = {
            key: value.copy()
            for key, value in world.dynamic_object_trajectories.items()
        }
        trajectories["context-object"][0, 0] += np.float32(0.5)
        world = replace(world, dynamic_object_trajectories=trajectories)
    elif tamper == "context_spec":
        specs = {key: dict(value) for key, value in world.dynamic_object_specs.items()}
        specs["context-object"] = {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.26},
        }
        world = replace(world, dynamic_object_specs=specs)
    elif tamper == "static_occupancy":
        static_occupancy = world.static_occupancy.copy()
        static_occupancy[0, 0] = np.float32(1.0)
        world = replace(world, static_occupancy=static_occupancy)
    elif tamper == "occluders":
        world = replace(
            world,
            occluders=(
                {
                    "kind": "synthetic_wall",
                    "polygon_xy": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.2]],
                    "height_m": 1.5,
                },
            ),
        )
    elif tamper == "structural_blind_spot_config":
        world = replace(
            world,
            blind_spot_config={
                "kind": "structural",
                "source": "tampered-but-valid",
            },
        )
    elif tamper == "random_seed":
        world = replace(world, random_seed=world.random_seed + 1)
    else:
        metadata = dict(world.metadata)
        metadata["nested_context"] = {"source": "tampered", "weight": 0.5}
        world = replace(world, metadata=metadata)
    save_dataclass(world, world_path)

    with pytest.raises(ValueError, match="world_semantic_digest mismatch"):
        module.load_event_target_motion_shard(output, grid=_grid())


def test_loader_expected_base_and_trajectory_sets_are_exact(tmp_path: Path) -> None:
    module = _sut()
    first = _record(module, "a")
    second = _record(module, "b")
    output = tmp_path / "expected-sets"
    _write(
        module,
        output,
        [first, second],
        [_world(first, _grid()), _world(second, _grid())],
    )
    with pytest.raises(ValueError, match="base_state_id set"):
        module.load_event_target_motion_shard(
            output, grid=_grid(), expected_base_state_ids={first.base_state_id}
        )
    with pytest.raises(ValueError, match="trajectory_id set"):
        module.load_event_target_motion_shard(
            output,
            grid=_grid(),
            expected_trajectory_ids={first.trajectory_id, "unknown"},
        )
    with pytest.raises(ValueError, match="generated_event_id set"):
        module.load_event_target_motion_shard(
            output,
            grid=_grid(),
            expected_generated_event_ids={first.generated_event_id},
        )
    with pytest.raises(ValueError, match="generated_event_id set"):
        module.load_event_target_motion_shard(
            output,
            grid=_grid(),
            expected_generated_event_ids={
                first.generated_event_id,
                second.generated_event_id,
                "unknown-event",
            },
        )


def test_writer_runs_formal_loader_before_publish_and_cleans_only_own_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    record = _record(module)
    output = tmp_path / "atomic"
    foreign = tmp_path / ".atomic.staging-foreign"
    foreign.mkdir()

    def fail_self_check(*args, **kwargs):
        assert kwargs["expected_generated_event_ids"] == {
            record.generated_event_id
        }
        raise ValueError("forced formal-loader failure")

    monkeypatch.setattr(module, "load_event_target_motion_shard", fail_self_check)
    with pytest.raises(ValueError, match="forced formal-loader failure"):
        _write(module, output, [record], [_world(record, _grid())])
    assert not output.exists()
    assert foreign.is_dir()
    assert [path for path in tmp_path.iterdir() if path != foreign] == []


def test_build_renderer_scene_copies_all_context_and_target_and_rejects_collision() -> None:
    module = _sut()
    record = _record(module)
    context_history = np.zeros((8, 3), dtype=np.float32)
    context_future = np.ones((15, 3), dtype=np.float32)
    context_spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.25},
    }
    context = OracleContext(
        base_state_id=record.base_state_id,
        dynamic_object_history={"context-object": context_history},
        dynamic_object_future={"context-object": context_future},
        dynamic_object_specs={"context-object": context_spec},
        metadata={},
    )
    scene = module.build_renderer_scene(record, context)
    assert tuple(scene.dynamic_object_history) == (
        "context-object",
        record.target_dynamic_object_id,
    )
    assert set(scene.dynamic_object_specs) == {
        "context-object",
        record.target_dynamic_object_id,
    }
    np.testing.assert_array_equal(
        scene.dynamic_object_history[record.target_dynamic_object_id],
        record.history_poses,
    )
    assert not np.shares_memory(
        scene.dynamic_object_history["context-object"], context_history
    )
    assert not np.shares_memory(
        scene.dynamic_object_history[record.target_dynamic_object_id],
        record.history_poses,
    )
    context_history[:] = 9.0
    assert not np.all(scene.dynamic_object_history["context-object"] == 9.0)

    collision = OracleContext(
        base_state_id=record.base_state_id,
        dynamic_object_history={record.target_dynamic_object_id: np.zeros((8, 3), dtype=np.float32)},
        dynamic_object_future={record.target_dynamic_object_id: np.zeros((15, 3), dtype=np.float32)},
        dynamic_object_specs={record.target_dynamic_object_id: context_spec},
        metadata={},
    )
    with pytest.raises(ValueError, match="target id collides"):
        module.build_renderer_scene(record, collision)
