"""Deterministic observation-consistent hidden-world scenario banks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from src.contracts import (
    ARRAY_DTYPE,
    SCHEMA_VERSION,
    GridSpec,
    OracleWorld,
    validate_oracle_world,
)
from src.geometry import rasterize_footprint
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.utils.seeding import derive_seed, stable_digest


SCENARIO_BANK_VERSION = "scenario_bank_v1"
SUPPORTED_BANK_SIZES = (8, 16, 32)
SCENARIO_VARIANT_KINDS = (
    "current",
    "empty",
    "temporal",
    "spatial",
    "speed",
    "irrelevant",
)
_TOP_KEYS = frozenset({"schema_version", "scenario_bank", "posterior", "decision"})
_SCENARIO_KEYS = frozenset(
    {
        "version",
        "compositions",
        "temporal_step_offsets",
        "spatial_offsets_m",
        "speed_scales",
        "irrelevant_offsets_m",
    }
)


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("scenario value must be finite canonical JSON") from exc


@dataclass(frozen=True)
class ScenarioBankConfig:
    version: str
    compositions: Mapping[int, Mapping[str, int]]
    temporal_step_offsets: tuple[int, ...]
    spatial_offsets_m: tuple[float, ...]
    speed_scales: tuple[float, ...]
    irrelevant_offsets_m: tuple[tuple[float, float], ...]
    digest: str

    def composition(self, size: int) -> dict[str, int]:
        requested = _integer(size, name="size")
        if requested not in SUPPORTED_BANK_SIZES:
            raise ValueError("scenario bank size must be 8, 16, or 32")
        return dict(self.compositions[requested])


@dataclass(frozen=True)
class ScenarioHypothesis:
    hypothesis_id: str
    variant_kind: str
    world: OracleWorld
    current_dynamic_poses: Mapping[str, np.ndarray]
    seed_namespace: str
    transform: Mapping[str, object]


@dataclass(frozen=True)
class ScenarioBank:
    version: str
    split: str
    source_namespace: str
    base_state_id: str
    target_object_id: str
    size: int
    composition: Mapping[str, int]
    current_visible_mask: np.ndarray
    current_visible_occupancy_digest: str
    hypotheses: tuple[ScenarioHypothesis, ...]
    config_digest: str
    semantic_digest: str


def _sequence(
    value: Any, *, name: str, converter
) -> tuple:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(converter(item, name=f"{name}[{index}]") for index, item in enumerate(value))


def load_scenario_bank_config(path: str | Path) -> ScenarioBankConfig:
    """Load the frozen M=8/16/32 composition and transform schedules."""

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid verification GT config: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _TOP_KEYS:
        raise ValueError("verification GT config keys are invalid")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"verification GT schema must be {SCHEMA_VERSION}")
    section = raw["scenario_bank"]
    if not isinstance(section, dict) or set(section) != _SCENARIO_KEYS:
        raise ValueError("scenario_bank config keys are invalid")
    if section["version"] != SCENARIO_BANK_VERSION:
        raise ValueError("unsupported scenario bank version")
    raw_compositions = section["compositions"]
    if not isinstance(raw_compositions, dict) or set(raw_compositions) != set(
        SUPPORTED_BANK_SIZES
    ):
        raise ValueError("scenario compositions must define 8, 16, and 32")
    compositions: dict[int, Mapping[str, int]] = {}
    for size in SUPPORTED_BANK_SIZES:
        value = raw_compositions[size]
        if not isinstance(value, dict) or set(value) != set(SCENARIO_VARIANT_KINDS):
            raise ValueError("scenario composition variant kinds are invalid")
        counts = {
            kind: _integer(value[kind], name=f"composition[{size}].{kind}")
            for kind in SCENARIO_VARIANT_KINDS
        }
        if any(count <= 0 for count in counts.values()) or sum(counts.values()) != size:
            raise ValueError(f"scenario composition {size} must contain positive counts summing to size")
        if counts["current"] != 1:
            raise ValueError("every scenario composition must contain exactly one current world")
        compositions[size] = MappingProxyType(counts)

    temporal = _sequence(
        section["temporal_step_offsets"], name="temporal_step_offsets", converter=_integer
    )
    if any(value == 0 for value in temporal) or len(set(temporal)) != len(temporal):
        raise ValueError("temporal offsets must be unique and nonzero")
    spatial = _sequence(
        section["spatial_offsets_m"], name="spatial_offsets_m", converter=_finite_real
    )
    if any(value == 0.0 for value in spatial) or len(set(spatial)) != len(spatial):
        raise ValueError("spatial offsets must be unique and nonzero")
    speeds = _sequence(
        section["speed_scales"], name="speed_scales", converter=_finite_real
    )
    if any(value <= 0.0 or value == 1.0 for value in speeds) or len(set(speeds)) != len(speeds):
        raise ValueError("speed scales must be positive, unique, and differ from one")
    raw_irrelevant = section["irrelevant_offsets_m"]
    if not isinstance(raw_irrelevant, list) or not raw_irrelevant:
        raise ValueError("irrelevant_offsets_m must be a non-empty list")
    irrelevant: list[tuple[float, float]] = []
    for index, pair in enumerate(raw_irrelevant):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("each irrelevant offset must contain x and y")
        offset = tuple(
            _finite_real(value, name=f"irrelevant_offsets_m[{index}]") for value in pair
        )
        if offset == (0.0, 0.0):
            raise ValueError("irrelevant offsets must be nonzero")
        irrelevant.append(offset)
    maximum = compositions[32]
    if (
        len(temporal) < maximum["temporal"]
        or len(spatial) < maximum["spatial"]
        or len(speeds) < maximum["speed"]
        or len(irrelevant) < maximum["irrelevant"]
    ):
        raise ValueError("scenario transform schedules are shorter than the M=32 preset")
    scientific = {
        "version": SCENARIO_BANK_VERSION,
        "compositions": {str(size): dict(compositions[size]) for size in SUPPORTED_BANK_SIZES},
        "temporal_step_offsets": list(temporal),
        "spatial_offsets_m": list(spatial),
        "speed_scales": list(speeds),
        "irrelevant_offsets_m": [list(value) for value in irrelevant],
    }
    return ScenarioBankConfig(
        version=SCENARIO_BANK_VERSION,
        compositions=MappingProxyType(compositions),
        temporal_step_offsets=temporal,
        spatial_offsets_m=spatial,
        speed_scales=speeds,
        irrelevant_offsets_m=tuple(irrelevant),
        digest=hashlib.sha256(_canonical_json(scientific).encode("utf-8")).hexdigest(),
    )


def _owned_current_poses(
    values: Mapping[str, np.ndarray], *, expected_ids: set[str]
) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping) or set(values) != expected_ids:
        raise ValueError("current dynamic pose IDs must align with world specs")
    copied: dict[str, np.ndarray] = {}
    for object_id in sorted(expected_ids):
        value = values[object_id]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (3,)
            or value.dtype != ARRAY_DTYPE
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"current pose for {object_id!r} must be finite float32 [3]")
        item = np.array(value, dtype=ARRAY_DTYPE, order="C", copy=True)
        item.setflags(write=False)
        copied[object_id] = item
    return MappingProxyType(copied)


def _copy_world(
    reference: OracleWorld,
    *,
    trajectories: Mapping[str, np.ndarray],
    specs: Mapping[str, dict[str, object]],
    world_id: str,
    random_seed: int,
    metadata: Mapping[str, object],
) -> OracleWorld:
    return OracleWorld(
        world_id=world_id,
        base_state_id=reference.base_state_id,
        static_occupancy=np.array(
            reference.static_occupancy, dtype=ARRAY_DTYPE, order="C", copy=True
        ),
        dynamic_object_trajectories={
            key: np.array(value, dtype=ARRAY_DTYPE, order="C", copy=True)
            for key, value in sorted(trajectories.items())
        },
        dynamic_object_specs={key: deepcopy(value) for key, value in sorted(specs.items())},
        occluders=tuple(deepcopy(reference.occluders)),
        blind_spot_config=deepcopy(reference.blind_spot_config),
        random_seed=int(random_seed),
        metadata=deepcopy(dict(metadata)),
    )


def _temporal_variant(
    current: np.ndarray, future: np.ndarray, *, offset_steps: int
) -> np.ndarray:
    source = np.vstack((current[None, :], future)).astype(np.float64)
    result = np.empty_like(future, dtype=np.float64)
    for output_index in range(future.shape[0]):
        source_index = int(np.clip(output_index + 1 + offset_steps, 0, future.shape[0]))
        result[output_index] = source[source_index]
    return result.astype(ARRAY_DTYPE)


def _spatial_variant(
    current: np.ndarray, future: np.ndarray, *, offset_m: float
) -> np.ndarray:
    normal = np.asarray([-np.sin(current[2]), np.cos(current[2])], dtype=np.float64)
    fractions = (
        np.arange(future.shape[0], dtype=np.float64) + 1.0
    ) / future.shape[0]
    result = future.astype(np.float64).copy()
    result[:, :2] += fractions[:, None] * offset_m * normal[None, :]
    return result.astype(ARRAY_DTYPE)


def _speed_variant(
    current: np.ndarray, future: np.ndarray, *, scale: float
) -> np.ndarray:
    source = np.vstack((current[None, :], future)).astype(np.float64)
    source_times = np.arange(source.shape[0], dtype=np.float64)
    query = np.clip(
        (np.arange(future.shape[0], dtype=np.float64) + 1.0) * scale,
        0.0,
        float(future.shape[0]),
    )
    unwrapped_yaw = np.unwrap(source[:, 2])
    result = np.column_stack(
        (
            np.interp(query, source_times, source[:, 0]),
            np.interp(query, source_times, source[:, 1]),
            np.interp(query, source_times, unwrapped_yaw),
        )
    )
    return result.astype(ARRAY_DTYPE)


def _irrelevant_variant(
    future: np.ndarray, *, offset_xy: tuple[float, float]
) -> np.ndarray:
    fractions = (
        np.arange(future.shape[0], dtype=np.float64) + 1.0
    ) / future.shape[0]
    result = future.astype(np.float64).copy()
    result[:, :2] += fractions[:, None] * np.asarray(offset_xy, dtype=np.float64)
    return result.astype(ARRAY_DTYPE)


def _array_digest(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(value.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _visible_occupancy(
    hypothesis: ScenarioHypothesis, *, visible_mask: np.ndarray, grid: GridSpec
) -> np.ndarray:
    occupancy = hypothesis.world.static_occupancy != 0.0
    for object_id in sorted(hypothesis.current_dynamic_poses):
        occupancy |= rasterize_footprint(
            footprint_from_spec(hypothesis.world.dynamic_object_specs[object_id]),
            hypothesis.current_dynamic_poses[object_id],
            grid,
        )
    return occupancy & visible_mask


def _bank_semantic_payload(bank: ScenarioBank) -> dict[str, object]:
    return {
        "version": bank.version,
        "split": bank.split,
        "source_namespace": bank.source_namespace,
        "base_state_id": bank.base_state_id,
        "target_object_id": bank.target_object_id,
        "size": bank.size,
        "composition": dict(bank.composition),
        "current_visible_mask": _array_digest(bank.current_visible_mask.astype(np.uint8)),
        "current_visible_occupancy_digest": bank.current_visible_occupancy_digest,
        "config_digest": bank.config_digest,
        "hypotheses": [
            {
                "hypothesis_id": item.hypothesis_id,
                "variant_kind": item.variant_kind,
                "seed_namespace": item.seed_namespace,
                "transform": dict(item.transform),
                "world_id": item.world.world_id,
                "random_seed": item.world.random_seed,
                "static": _array_digest(item.world.static_occupancy),
                "future": {
                    key: _array_digest(value)
                    for key, value in sorted(item.world.dynamic_object_trajectories.items())
                },
                "specs": item.world.dynamic_object_specs,
                "current": {
                    key: _array_digest(value)
                    for key, value in sorted(item.current_dynamic_poses.items())
                },
            }
            for item in bank.hypotheses
        ],
    }


def _bank_digest(bank: ScenarioBank) -> str:
    return hashlib.sha256(
        _canonical_json(_bank_semantic_payload(bank)).encode("utf-8")
    ).hexdigest()


def _validate_split_namespace(split: str, source_namespace: str) -> None:
    if split not in {"train", "calibration", "val", "test"}:
        raise ValueError("split is invalid")
    _nonempty_string(source_namespace, name="source_namespace")
    if f"/{split}/" not in f"/{source_namespace}/":
        raise ValueError("source namespace must bind the declared split")


def validate_scenario_bank(bank: ScenarioBank, *, grid: GridSpec) -> None:
    """Prove observation consistency and target-only variation, fail closed."""

    if not isinstance(bank, ScenarioBank):
        raise TypeError("bank must be a ScenarioBank")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if bank.version != SCENARIO_BANK_VERSION:
        raise ValueError("unsupported scenario bank version")
    _validate_split_namespace(bank.split, bank.source_namespace)
    if bank.size not in SUPPORTED_BANK_SIZES or len(bank.hypotheses) != bank.size:
        raise ValueError("scenario bank size must be 8, 16, or 32")
    if set(bank.composition) != set(SCENARIO_VARIANT_KINDS) or sum(
        bank.composition.values()
    ) != bank.size:
        raise ValueError("scenario bank composition is invalid")
    observed_counts = Counter(item.variant_kind for item in bank.hypotheses)
    if observed_counts != Counter(dict(bank.composition)):
        raise ValueError("scenario bank hypothesis composition mismatch")
    if bank.hypotheses[0].variant_kind != "current":
        raise ValueError("current scenario must be first")
    visible_mask = np.asarray(bank.current_visible_mask)
    if visible_mask.shape != (grid.height, grid.width) or visible_mask.dtype != np.bool_:
        raise ValueError("current visible mask must be bool with grid shape")
    if len({item.hypothesis_id for item in bank.hypotheses}) != bank.size:
        raise ValueError("scenario hypothesis IDs must be unique")
    if len({item.seed_namespace for item in bank.hypotheses}) != bank.size:
        raise ValueError("scenario seed namespaces must be unique")
    if any(
        not item.seed_namespace.startswith(f"scenario/{bank.split}/")
        for item in bank.hypotheses
    ):
        raise ValueError("scenario seed namespace split mismatch")

    reference = bank.hypotheses[0]
    validate_oracle_world(reference.world, grid)
    if reference.world.base_state_id != bank.base_state_id:
        raise ValueError("scenario base_state_id mismatch")
    reference_ids = set(reference.world.dynamic_object_trajectories)
    if bank.target_object_id not in reference_ids:
        raise ValueError("target object is absent from current scenario")
    reference_current = _owned_current_poses(
        reference.current_dynamic_poses, expected_ids=reference_ids
    )
    target_mask = rasterize_footprint(
        footprint_from_spec(reference.world.dynamic_object_specs[bank.target_object_id]),
        reference_current[bank.target_object_id],
        grid,
    )
    if np.any(target_mask & visible_mask):
        raise ValueError("scenario target must be hidden in current visible occupancy")
    non_target_ids = reference_ids - {bank.target_object_id}
    reference_static = reference.world.static_occupancy
    if not np.isin(reference_static, (0.0, 1.0)).all():
        raise ValueError("scenario static occupancy must be binary")
    expected_visible = _visible_occupancy(reference, visible_mask=visible_mask, grid=grid)
    expected_visible_digest = _array_digest(expected_visible.astype(np.uint8))
    if bank.current_visible_occupancy_digest != expected_visible_digest:
        raise ValueError("current visible occupancy digest mismatch")

    for hypothesis in bank.hypotheses:
        _nonempty_string(hypothesis.hypothesis_id, name="hypothesis_id")
        if hypothesis.variant_kind not in SCENARIO_VARIANT_KINDS:
            raise ValueError("scenario variant kind is invalid")
        validate_oracle_world(hypothesis.world, grid)
        if hypothesis.world.base_state_id != bank.base_state_id:
            raise ValueError("scenario base_state_id mismatch")
        if not np.array_equal(hypothesis.world.static_occupancy, reference_static):
            raise ValueError("scenario static occupancy differs across worlds")
        ids = set(hypothesis.world.dynamic_object_trajectories)
        current = _owned_current_poses(
            hypothesis.current_dynamic_poses, expected_ids=ids
        )
        target_present = bank.target_object_id in ids
        if hypothesis.variant_kind == "empty":
            if target_present:
                raise ValueError("empty scenario must remove only the target")
        elif not target_present:
            raise ValueError("only empty scenarios may remove the target")
        if ids - {bank.target_object_id} != non_target_ids:
            raise ValueError("scenario non-target object set changed")
        for object_id in sorted(non_target_ids):
            if (
                hypothesis.world.dynamic_object_specs[object_id]
                != reference.world.dynamic_object_specs[object_id]
                or not np.array_equal(
                    hypothesis.world.dynamic_object_trajectories[object_id],
                    reference.world.dynamic_object_trajectories[object_id],
                )
                or not np.array_equal(current[object_id], reference_current[object_id])
            ):
                raise ValueError("scenario non-target object changed")
        if target_present and hypothesis.world.dynamic_object_specs[
            bank.target_object_id
        ] != reference.world.dynamic_object_specs[bank.target_object_id]:
            raise ValueError("scenario target footprint/type changed")
        visible = _visible_occupancy(hypothesis, visible_mask=visible_mask, grid=grid)
        if not np.array_equal(visible, expected_visible):
            raise ValueError("scenario current visible occupancy differs across worlds")
        for object_id in sorted(ids):
            footprint = footprint_from_spec(hypothesis.world.dynamic_object_specs[object_id])
            for pose in hypothesis.world.dynamic_object_trajectories[object_id]:
                if np.any(
                    rasterize_footprint(footprint, pose, grid)
                    & (reference_static != 0.0)
                ):
                    raise ValueError("scenario dynamic trajectory violates static geometry")
    if _bank_digest(bank) != bank.semantic_digest:
        raise ValueError("scenario bank semantic digest mismatch")


def build_scenario_bank(
    *,
    current_world: OracleWorld,
    target_object_id: str,
    current_dynamic_poses: Mapping[str, np.ndarray],
    current_visible_mask: np.ndarray,
    grid: GridSpec,
    split: str,
    source_namespace: str,
    seed: int,
    size: int,
    config: ScenarioBankConfig,
) -> ScenarioBank:
    """Generate deterministic target-only variants from one hidden current world."""

    if not isinstance(current_world, OracleWorld):
        raise TypeError("current_world must be an OracleWorld")
    if not isinstance(config, ScenarioBankConfig):
        raise TypeError("config must be a ScenarioBankConfig")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    validate_oracle_world(current_world, grid)
    target_id = _nonempty_string(target_object_id, name="target_object_id")
    object_ids = set(current_world.dynamic_object_trajectories)
    if target_id not in object_ids:
        raise ValueError("target_object_id is absent from current_world")
    current = _owned_current_poses(current_dynamic_poses, expected_ids=object_ids)
    _validate_split_namespace(split, source_namespace)
    seed_value = _integer(seed, name="seed")
    composition = config.composition(size)
    if (
        not isinstance(current_visible_mask, np.ndarray)
        or current_visible_mask.shape != (grid.height, grid.width)
        or current_visible_mask.dtype != np.bool_
    ):
        raise ValueError("current_visible_mask must be bool with grid shape")
    visible_mask = np.array(current_visible_mask, dtype=bool, order="C", copy=True)
    visible_mask.setflags(write=False)

    target_future = current_world.dynamic_object_trajectories[target_id]
    target_current = current[target_id]
    schedules: dict[str, Sequence[object]] = {
        "current": (None,),
        "empty": tuple(range(composition["empty"])),
        "temporal": config.temporal_step_offsets[: composition["temporal"]],
        "spatial": config.spatial_offsets_m[: composition["spatial"]],
        "speed": config.speed_scales[: composition["speed"]],
        "irrelevant": config.irrelevant_offsets_m[: composition["irrelevant"]],
    }
    hypotheses: list[ScenarioHypothesis] = []
    for variant_kind in SCENARIO_VARIANT_KINDS:
        for variant_index, parameter in enumerate(schedules[variant_kind]):
            trajectories = {
                key: np.array(value, dtype=ARRAY_DTYPE, order="C", copy=True)
                for key, value in current_world.dynamic_object_trajectories.items()
            }
            specs = deepcopy(current_world.dynamic_object_specs)
            current_poses = {
                key: np.array(value, dtype=ARRAY_DTYPE, order="C", copy=True)
                for key, value in current.items()
            }
            if variant_kind == "empty":
                trajectories.pop(target_id)
                specs.pop(target_id)
                current_poses.pop(target_id)
                transform = {"kind": "empty", "replicate_index": variant_index}
            elif variant_kind == "temporal":
                trajectories[target_id] = _temporal_variant(
                    target_current, target_future, offset_steps=int(parameter)
                )
                transform = {"kind": "temporal", "offset_steps": int(parameter)}
            elif variant_kind == "spatial":
                trajectories[target_id] = _spatial_variant(
                    target_current, target_future, offset_m=float(parameter)
                )
                transform = {"kind": "spatial", "offset_m": float(parameter)}
            elif variant_kind == "speed":
                trajectories[target_id] = _speed_variant(
                    target_current, target_future, scale=float(parameter)
                )
                transform = {"kind": "speed", "scale": float(parameter)}
            elif variant_kind == "irrelevant":
                offset_xy = tuple(float(value) for value in parameter)
                trajectories[target_id] = _irrelevant_variant(
                    target_future, offset_xy=offset_xy
                )
                transform = {"kind": "irrelevant", "offset_xy_m": list(offset_xy)}
            else:
                transform = {"kind": "current"}
            namespace = (
                f"scenario/{split}/seed-{seed_value}/"
                f"{variant_kind}-{variant_index:02d}"
            )
            hypothesis_seed = derive_seed(
                seed_value,
                SCENARIO_BANK_VERSION,
                current_world.world_id,
                source_namespace,
                size,
                variant_kind,
                variant_index,
                _canonical_json(transform),
            )
            identity = stable_digest(
                SCENARIO_BANK_VERSION,
                current_world.world_id,
                target_id,
                namespace,
                config.digest,
                size,
                size=16,
            )
            hypothesis_id = f"scenario-{identity}"
            world_id = f"scenario-world-{identity}"
            metadata = {
                **deepcopy(current_world.metadata),
                "scenario_bank_version": SCENARIO_BANK_VERSION,
                "scenario_variant_kind": variant_kind,
                "scenario_variant_index": variant_index,
                "scenario_seed_namespace": namespace,
                "scenario_source_namespace": source_namespace,
                "scenario_split": split,
                "scenario_target_object_id": target_id,
                "scenario_transform": transform,
            }
            world = _copy_world(
                current_world,
                trajectories=trajectories,
                specs=specs,
                world_id=world_id,
                random_seed=hypothesis_seed,
                metadata=metadata,
            )
            hypotheses.append(
                ScenarioHypothesis(
                    hypothesis_id=hypothesis_id,
                    variant_kind=variant_kind,
                    world=world,
                    current_dynamic_poses=_owned_current_poses(
                        current_poses, expected_ids=set(trajectories)
                    ),
                    seed_namespace=namespace,
                    transform=MappingProxyType(deepcopy(transform)),
                )
            )
    provisional = ScenarioBank(
        version=SCENARIO_BANK_VERSION,
        split=split,
        source_namespace=source_namespace,
        base_state_id=current_world.base_state_id,
        target_object_id=target_id,
        size=size,
        composition=MappingProxyType(dict(composition)),
        current_visible_mask=visible_mask,
        current_visible_occupancy_digest="pending",
        hypotheses=tuple(hypotheses),
        config_digest=config.digest,
        semantic_digest="pending",
    )
    visible_digest = _array_digest(
        _visible_occupancy(
            provisional.hypotheses[0], visible_mask=visible_mask, grid=grid
        ).astype(np.uint8)
    )
    provisional = ScenarioBank(
        **{
            **provisional.__dict__,
            "current_visible_occupancy_digest": visible_digest,
        }
    )
    bank = ScenarioBank(
        **{
            **provisional.__dict__,
            "semantic_digest": _bank_digest(provisional),
        }
    )
    validate_scenario_bank(bank, grid=grid)
    return bank


__all__ = (
    "SCENARIO_BANK_VERSION",
    "SCENARIO_VARIANT_KINDS",
    "SUPPORTED_BANK_SIZES",
    "ScenarioBank",
    "ScenarioBankConfig",
    "ScenarioHypothesis",
    "build_scenario_bank",
    "load_scenario_bank_config",
    "validate_scenario_bank",
)
