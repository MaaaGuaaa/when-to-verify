"""SOP09 R0/R1 trajectory-conditioned risk models and checkpoint v2."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from src.contracts import (
    N_HISTORY_CHANNELS,
    N_STATE_CHANNELS,
    N_TRAJECTORY_CHANNELS,
    QUANTILE_LEVELS,
    ROBOT_STATE_DIM,
    SCHEMA_VERSION,
)
from src.datasets.risk_dataloader import (
    MODEL_INPUT_KEYS,
    RiskDataContractError,
    validate_model_input_mapping,
)
from src.datasets.toy_risk_learning import (
    ToyRiskDataset,
    assert_toy_split_isolation,
    frozen_channel_spec,
    validate_toy_risk_dataset_publication,
)
from src.models.bev_encoder import BEVEncoder, ConvGRUCell
from src.models.losses import risk_loss

RISK_CHECKPOINT_LAYOUT_VERSION = "risk_model_checkpoint_v2"
RISK_MODEL_VARIANTS: tuple[str, ...] = ("r0", "r1")
TRAJECTORY_SENSITIVITY_EPSILON = 1e-8
RISK_COMMON_PROVENANCE_KEYS = frozenset(
    {"schema_version", "channel_spec", "model_variant", "config_digest", "seed"}
)
RISK_TOY_PROVENANCE_KEYS = frozenset(
    {
        *RISK_COMMON_PROVENANCE_KEYS,
        "toy_dataset_manifest_digest",
        "validation_dataset_manifest_digest",
    }
)
RISK_PRODUCTION_PROVENANCE_KEYS = frozenset(
    {
        *RISK_COMMON_PROVENANCE_KEYS,
        "g1_split_manifest_digest",
        "risk_dataset_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
    }
)
RISK_CHECKPOINT_TOP_LEVEL_KEYS = frozenset(
    {
        "checkpoint_layout_version",
        "mode",
        "model_config",
        "model_state_dict",
        "model_state_digest_sha256",
        "provenance",
        "inference_parameters",
        "checkpoint_semantic_digest_sha256",
    }
)
RISK_INFERENCE_PARAMETER_KEYS = frozenset(
    {"quantile_levels", "collision_probability"}
)


def noncrossing_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """Map four unconstrained values to monotone quantiles in ``(0, 1)``."""

    if raw.ndim != 2 or raw.shape[1] != len(QUANTILE_LEVELS):
        raise ValueError("raw quantiles must have shape [B,4]")
    first = torch.sigmoid(raw[:, :1])
    values = [first]
    previous = first
    for index in range(1, raw.shape[1]):
        fraction = torch.sigmoid(raw[:, index : index + 1])
        previous = previous + (1.0 - previous) * fraction
        values.append(previous)
    return torch.cat(values, dim=1)


class RiskModel(nn.Module):
    """R0 stacked-history CNN or R1 ConvGRU temporal risk model."""

    def __init__(
        self,
        *,
        variant: str = "r0",
        hidden_channels: int = 16,
        history_steps: int = 8,
    ) -> None:
        super().__init__()
        if variant not in RISK_MODEL_VARIANTS:
            raise ValueError(f"variant must be one of {RISK_MODEL_VARIANTS}")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if history_steps != 8:
            raise ValueError("SOP09 frozen history_steps must equal 8")
        self.variant = variant
        self.hidden_channels = int(hidden_channels)
        self.history_steps = int(history_steps)
        if variant == "r0":
            input_channels = (
                history_steps * N_HISTORY_CHANNELS
                + N_STATE_CHANNELS
                + N_TRAJECTORY_CHANNELS
            )
            self.spatial_encoder = BEVEncoder(input_channels, hidden_channels)
            fused_features = hidden_channels + ROBOT_STATE_DIM
        else:
            self.history_encoder = BEVEncoder(N_HISTORY_CHANNELS, hidden_channels)
            self.temporal_cell = ConvGRUCell(hidden_channels, hidden_channels)
            self.context_encoder = BEVEncoder(
                N_STATE_CHANNELS + N_TRAJECTORY_CHANNELS, hidden_channels
            )
            fused_features = 2 * hidden_channels + ROBOT_STATE_DIM
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(
            nn.Linear(fused_features, 2 * hidden_channels),
            nn.ReLU(inplace=False),
        )
        self.quantile_head = nn.Linear(2 * hidden_channels, len(QUANTILE_LEVELS))
        self.collision_head = nn.Linear(2 * hidden_channels, 1)

    def export_config(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "hidden_channels": self.hidden_channels,
            "history_steps": self.history_steps,
        }

    def _validate_inputs(self, inputs: Mapping[str, torch.Tensor]) -> None:
        if not isinstance(inputs, Mapping):
            raise RiskDataContractError("risk model inputs must be a mapping")
        if set(inputs) != set(MODEL_INPUT_KEYS):
            raise RiskDataContractError(
                f"risk model input keys must be exactly {MODEL_INPUT_KEYS}"
            )
        validate_model_input_mapping(inputs)
        history = inputs["bev_history"]
        state = inputs["state_channels"]
        trajectory = inputs["trajectory_channels"]
        robot_state = inputs["robot_state"]
        if history.ndim != 5 or history.shape[1:3] != (
            self.history_steps,
            N_HISTORY_CHANNELS,
        ):
            raise RiskDataContractError(
                "bev_history must have frozen history shape [B,8,2,H,W]"
            )
        if state.ndim != 4 or state.shape[1] != N_STATE_CHANNELS:
            raise RiskDataContractError("state_channels must have shape [B,9,H,W]")
        if trajectory.ndim != 4 or trajectory.shape[1] != N_TRAJECTORY_CHANNELS:
            raise RiskDataContractError(
                "trajectory_channels must have shape [B,4,H,W]"
            )
        if robot_state.ndim != 2 or robot_state.shape[1] != ROBOT_STATE_DIM:
            raise RiskDataContractError("robot_state must have shape [B,2]")
        batch = history.shape[0]
        if any(value.shape[0] != batch for value in (state, trajectory, robot_state)):
            raise RiskDataContractError("all risk model inputs must share batch size")
        spatial = history.shape[-2:]
        if state.shape[-2:] != spatial or trajectory.shape[-2:] != spatial:
            raise RiskDataContractError("all spatial risk model inputs must share H,W")

    def forward(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(inputs)
        history = inputs["bev_history"]
        state = inputs["state_channels"]
        trajectory = inputs["trajectory_channels"]
        robot_state = inputs["robot_state"]
        batch, steps, channels, height, width = history.shape
        if self.variant == "r0":
            spatial = torch.cat(
                (history.reshape(batch, steps * channels, height, width), state, trajectory),
                dim=1,
            )
            features = self.pool(self.spatial_encoder(spatial)).flatten(1)
        else:
            hidden: torch.Tensor | None = None
            for step in range(steps):
                encoded = self.history_encoder(history[:, step])
                hidden = self.temporal_cell(encoded, hidden)
            assert hidden is not None
            temporal_features = self.pool(hidden).flatten(1)
            context = self.context_encoder(torch.cat((state, trajectory), dim=1))
            context_features = self.pool(context).flatten(1)
            features = torch.cat((temporal_features, context_features), dim=1)
        fused = self.fusion(torch.cat((features, robot_state), dim=1))
        raw_quantiles = self.quantile_head(fused)
        collision_logits = self.collision_head(fused).squeeze(-1)
        return {
            "quantiles": noncrossing_quantiles(raw_quantiles),
            "collision_logits": collision_logits,
            "p_collision": torch.sigmoid(collision_logits),
        }


def _validate_provenance(mode: str, provenance: Mapping[str, object]) -> None:
    if mode not in {"toy", "production"}:
        raise RiskDataContractError("checkpoint mode must be toy or production")
    if not isinstance(provenance, Mapping):
        raise RiskDataContractError("checkpoint provenance must be a mapping")
    expected_keys = (
        RISK_TOY_PROVENANCE_KEYS
        if mode == "toy"
        else RISK_PRODUCTION_PROVENANCE_KEYS
    )
    if set(provenance) != expected_keys:
        missing = sorted(expected_keys - set(provenance))
        unexpected = sorted(repr(key) for key in set(provenance) - expected_keys)
        raise RiskDataContractError(
            "checkpoint provenance keys must match the mode-specific contract; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if provenance["schema_version"] != SCHEMA_VERSION:
        raise RiskDataContractError(
            f"checkpoint schema_version must be {SCHEMA_VERSION}"
        )
    if provenance["channel_spec"] != frozen_channel_spec():
        raise RiskDataContractError("checkpoint channel_spec mismatch")
    if provenance["model_variant"] not in RISK_MODEL_VARIANTS:
        raise RiskDataContractError("checkpoint model_variant mismatch")
    if not isinstance(provenance["seed"], int) or isinstance(
        provenance["seed"], bool
    ):
        raise RiskDataContractError("checkpoint seed must be an integer")
    string_fields = expected_keys - {
        "schema_version",
        "channel_spec",
        "model_variant",
        "seed",
    }
    for field in string_fields:
        value = provenance[field]
        if not isinstance(value, str) or not value:
            raise RiskDataContractError(f"checkpoint provenance {field} must be non-empty")
    if mode == "toy":
        for field in (
            "toy_dataset_manifest_digest",
            "validation_dataset_manifest_digest",
        ):
            value = provenance[field]
            if len(value) != 32 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise RiskDataContractError(
                    f"checkpoint provenance {field} must be a lowercase "
                    "BLAKE2b-128 digest"
                )
        if provenance["toy_dataset_manifest_digest"] == provenance[
            "validation_dataset_manifest_digest"
        ]:
            raise RiskDataContractError(
                "toy training and validation dataset manifest digests must be distinct"
            )


def _model_state_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash sorted tensor name/dtype/shape/content for tamper detection."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise RiskDataContractError("model_state_dict values must be tensors")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(size) for size in value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_semantic_digest(payload: Mapping[str, object]) -> str:
    """Bind model configuration, state digest, provenance, and inference API."""

    semantic = {
        "checkpoint_layout_version": payload.get("checkpoint_layout_version"),
        "mode": payload.get("mode"),
        "model_config": payload.get("model_config"),
        "model_state_digest_sha256": payload.get("model_state_digest_sha256"),
        "provenance": payload.get("provenance"),
        "inference_parameters": payload.get("inference_parameters"),
    }
    try:
        encoded = json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RiskDataContractError(
            f"checkpoint semantic payload is not finite JSON-safe data: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _validate_inference_parameters(value: object) -> None:
    if not isinstance(value, Mapping):
        raise RiskDataContractError(
            "checkpoint inference_parameters must be a mapping"
        )
    if set(value) != RISK_INFERENCE_PARAMETER_KEYS:
        raise RiskDataContractError(
            "checkpoint inference_parameters keys must be exactly "
            f"{sorted(RISK_INFERENCE_PARAMETER_KEYS)}"
        )
    quantile_levels = value["quantile_levels"]
    expected_quantile_levels = list(QUANTILE_LEVELS)
    valid_quantile_levels = (
        isinstance(quantile_levels, list)
        and len(quantile_levels) == len(expected_quantile_levels)
        and all(
            type(actual) in {int, float} and actual == expected
            for actual, expected in zip(quantile_levels, expected_quantile_levels)
        )
    )
    if not valid_quantile_levels:
        raise RiskDataContractError(
            "checkpoint inference_parameters quantile_levels must exactly equal "
            f"{expected_quantile_levels}"
        )
    if value["collision_probability"] != "sigmoid_logit":
        raise RiskDataContractError(
            "checkpoint inference_parameters collision_probability must equal "
            "'sigmoid_logit'"
        )


def save_risk_checkpoint(
    path: str | Path,
    *,
    model: RiskModel,
    mode: str,
    provenance: Mapping[str, object],
    inference_parameters: Mapping[str, object] | None = None,
) -> Path:
    """Atomically write a mode-bound checkpoint-v2 payload."""

    if not isinstance(model, RiskModel):
        raise TypeError("model must be RiskModel")
    _validate_provenance(mode, provenance)
    if provenance["model_variant"] != model.variant:
        raise RiskDataContractError("provenance model_variant does not match model")
    if inference_parameters is None:
        frozen_inference_parameters: object = {
            "quantile_levels": list(QUANTILE_LEVELS),
            "collision_probability": "sigmoid_logit",
        }
    elif isinstance(inference_parameters, Mapping):
        frozen_inference_parameters = copy.deepcopy(dict(inference_parameters))
    else:
        frozen_inference_parameters = inference_parameters
    _validate_inference_parameters(frozen_inference_parameters)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    model_state_dict = model.state_dict()
    payload = {
        "checkpoint_layout_version": RISK_CHECKPOINT_LAYOUT_VERSION,
        "mode": mode,
        "model_config": model.export_config(),
        "model_state_dict": model_state_dict,
        "model_state_digest_sha256": _model_state_digest(model_state_dict),
        "provenance": copy.deepcopy(dict(provenance)),
        "inference_parameters": frozen_inference_parameters,
    }
    payload["checkpoint_semantic_digest_sha256"] = _checkpoint_semantic_digest(
        payload
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_risk_checkpoint(
    path: str | Path,
    *,
    expected_mode: str,
    expected_provenance: Mapping[str, object] | None = None,
) -> tuple[RiskModel, dict[str, object]]:
    """Load a checkpoint only after version/mode/provenance validation."""

    try:
        payload = torch.load(Path(path), map_location="cpu")
    except (OSError, RuntimeError) as error:
        raise RiskDataContractError(f"unable to load risk checkpoint: {error}") from error
    if not isinstance(payload, dict):
        raise RiskDataContractError("risk checkpoint payload must be a mapping")
    if payload.get("checkpoint_layout_version") != RISK_CHECKPOINT_LAYOUT_VERSION:
        raise RiskDataContractError(
            f"checkpoint_layout_version must be {RISK_CHECKPOINT_LAYOUT_VERSION}"
        )
    if set(payload) != RISK_CHECKPOINT_TOP_LEVEL_KEYS:
        missing = sorted(RISK_CHECKPOINT_TOP_LEVEL_KEYS - set(payload))
        unexpected = sorted(
            repr(key) for key in set(payload) - RISK_CHECKPOINT_TOP_LEVEL_KEYS
        )
        raise RiskDataContractError(
            "risk checkpoint top-level keys must match the frozen contract; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if payload.get("mode") != expected_mode:
        raise RiskDataContractError(
            f"checkpoint mode mismatch: expected {expected_mode!r}, got {payload.get('mode')!r}"
        )
    provenance = payload.get("provenance")
    _validate_provenance(expected_mode, provenance)
    if expected_provenance is not None:
        _validate_provenance(expected_mode, expected_provenance)
        for field in sorted(expected_provenance):
            if provenance[field] != expected_provenance[field]:
                raise RiskDataContractError(
                    f"checkpoint provenance mismatch for {field}"
                )
    state_dict = payload.get("model_state_dict")
    expected_state_digest = payload.get("model_state_digest_sha256")
    if (
        not isinstance(expected_state_digest, str)
        or len(expected_state_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_state_digest)
    ):
        raise RiskDataContractError(
            "checkpoint model_state_digest_sha256 is missing or malformed"
        )
    if not isinstance(state_dict, Mapping):
        raise RiskDataContractError("checkpoint model_state_dict must be a mapping")
    actual_state_digest = _model_state_digest(state_dict)
    if not hmac.compare_digest(expected_state_digest, actual_state_digest):
        raise RiskDataContractError("checkpoint model_state_digest_sha256 mismatch")
    expected_semantic_digest = payload.get("checkpoint_semantic_digest_sha256")
    if (
        not isinstance(expected_semantic_digest, str)
        or len(expected_semantic_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_semantic_digest
        )
    ):
        raise RiskDataContractError(
            "checkpoint checkpoint_semantic_digest_sha256 is missing or malformed"
        )
    actual_semantic_digest = _checkpoint_semantic_digest(payload)
    if not hmac.compare_digest(expected_semantic_digest, actual_semantic_digest):
        raise RiskDataContractError("checkpoint checkpoint_semantic_digest_sha256 mismatch")
    _validate_inference_parameters(payload.get("inference_parameters"))
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise RiskDataContractError("checkpoint model_config must be a mapping")
    if model_config.get("variant") != provenance["model_variant"]:
        raise RiskDataContractError(
            "checkpoint model_config.variant does not match provenance.model_variant"
        )
    try:
        model = RiskModel(**model_config)
        model.load_state_dict(state_dict, strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise RiskDataContractError(f"invalid risk checkpoint model payload: {error}") from error
    model.eval()
    return model, payload


def _batch_loss(
    model: RiskModel,
    batch: object,
    *,
    lambda_collision: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    output = model(batch.model_inputs)
    losses = risk_loss(
        output,
        risk_severity=batch.targets["risk_severity"],
        collision_label=batch.targets["collision_label"],
        lambda_collision=lambda_collision,
    )
    return output, losses


def trajectory_ablation_sensitivity(
    model: RiskModel,
    batch: object,
    *,
    split: str,
) -> dict[str, object]:
    """Measure conditioning with a legal, label-free validation-query permutation.

    Every counterfactual query is another complete, validated validation-row
    query: its trajectory channels and corresponding robot state move together.
    History and current scene state remain fixed.  This diagnostic establishes
    conditioning only; it is not a directional real-world performance claim.
    """

    if split != "val":
        raise RiskDataContractError(
            "trajectory ablation diagnostic must use the validation split"
        )
    if getattr(batch, "split", None) != split:
        raise RiskDataContractError("trajectory ablation batch split mismatch")
    model_inputs = getattr(batch, "model_inputs", None)
    if not isinstance(model_inputs, Mapping):
        raise RiskDataContractError("trajectory ablation requires model_inputs")
    provenance = getattr(batch, "provenance", None)
    sample_ids = tuple(getattr(batch, "sample_ids", ()))
    if not isinstance(provenance, Mapping) or provenance.get("mode") != "toy":
        raise RiskDataContractError(
            "trajectory sensitivity requires validated toy provenance"
        )
    dataset_digest = provenance.get("toy_dataset_manifest_digest")
    rows_digest = provenance.get("manifest_rows_digest_sha256")
    if not isinstance(dataset_digest, str) or len(dataset_digest) != 32:
        raise RiskDataContractError("trajectory sensitivity dataset digest missing")
    if not isinstance(rows_digest, str) or len(rows_digest) != 64:
        raise RiskDataContractError("trajectory sensitivity row digest missing")
    trajectory = model_inputs["trajectory_channels"]
    robot_state = model_inputs["robot_state"]
    sample_count = len(sample_ids)
    if sample_count < 2 or trajectory.shape[0] != sample_count or (
        robot_state.shape[0] != sample_count
    ):
        raise RiskDataContractError(
            "trajectory sensitivity requires at least two aligned validation rows"
        )

    indices = torch.arange(sample_count, dtype=torch.long)
    best_permutation: torch.Tensor | None = None
    best_shift: int | None = None
    best_changed_count = -1
    for shift in range(1, sample_count):
        permutation = torch.roll(indices, shifts=-shift)
        candidate_trajectory = trajectory[permutation]
        candidate_robot_state = robot_state[permutation]
        trajectory_changed = torch.any(
            (candidate_trajectory != trajectory).reshape(sample_count, -1), dim=1
        )
        robot_state_changed = torch.any(
            (candidate_robot_state != robot_state).reshape(sample_count, -1), dim=1
        )
        changed_count = int(
            torch.count_nonzero(trajectory_changed | robot_state_changed).item()
        )
        if changed_count > best_changed_count:
            best_permutation = permutation
            best_shift = shift
            best_changed_count = changed_count
    if best_permutation is None or best_shift is None or best_changed_count < 1:
        raise RiskDataContractError(
            "validated rows do not provide a changed legal trajectory query"
        )
    counterfactual_inputs = {
        **model_inputs,
        "trajectory_channels": trajectory[best_permutation],
        "robot_state": robot_state[best_permutation],
    }
    was_training = model.training
    model.eval()
    with torch.no_grad():
        reference = model(model_inputs)
        counterfactual = model(counterfactual_inputs)
    if was_training:
        model.train()
    quantile_delta = float(
        torch.mean(
            torch.abs(reference["quantiles"] - counterfactual["quantiles"])
        ).item()
    )
    collision_logit_delta = float(
        torch.mean(
            torch.abs(
                reference["collision_logits"]
                - counterfactual["collision_logits"]
            )
        ).item()
    )
    collision_probability_delta = float(
        torch.mean(
            torch.abs(
                reference["p_collision"] - counterfactual["p_collision"]
            )
        ).item()
    )
    combined_delta = (
        quantile_delta + collision_logit_delta + collision_probability_delta
    )
    values = (
        quantile_delta,
        collision_logit_delta,
        collision_probability_delta,
        combined_delta,
    )
    if not all(torch.isfinite(torch.tensor(value)).item() for value in values):
        raise RiskDataContractError("trajectory ablation sensitivity is not finite")
    permutation_indices = [int(value) for value in best_permutation.tolist()]
    permutation_sample_ids = [sample_ids[index] for index in permutation_indices]
    permutation_digest = hashlib.sha256(
        json.dumps(
            permutation_sample_ids,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "protocol": "deterministic_permutation_of_validated_query",
        "diagnostic_kind": "legal_counterfactual_query_conditioning_sensitivity",
        "split": split,
        "sample_count": sample_count,
        "source_dataset_manifest_digest": dataset_digest,
        "source_manifest_rows_digest_sha256": rows_digest,
        "source_rows_strictly_validated": True,
        "query_components_permuted": ["trajectory_channels", "robot_state"],
        "permutation_shift": best_shift,
        "permutation_source_indices": permutation_indices,
        "permutation_source_sample_ids": permutation_sample_ids,
        "permutation_digest_sha256": permutation_digest,
        "changed_query_count": best_changed_count,
        "unchanged_query_count": sample_count - best_changed_count,
        "labels_accessed": False,
        "used_for_training_or_selection": False,
        "quantile_mean_absolute_delta": quantile_delta,
        "collision_logit_mean_absolute_delta": collision_logit_delta,
        "collision_probability_mean_absolute_delta": collision_probability_delta,
        "combined_mean_absolute_delta": combined_delta,
        "materiality_threshold": TRAJECTORY_SENSITIVITY_EPSILON,
        "materially_sensitive": combined_delta > TRAJECTORY_SENSITIVITY_EPSILON,
        "interpretation": (
            "conditioning_effect_only;"
            "does_not_establish_real_world_directional_superiority"
        ),
    }


def train_toy_risk_model(
    *,
    variant: str,
    train_dataset: ToyRiskDataset,
    validation_dataset: ToyRiskDataset,
    hidden_channels: int = 8,
    optimization_steps: int = 40,
    learning_rate: float = 0.02,
    lambda_collision: float = 1.0,
    seed: int = 42,
) -> tuple[RiskModel, dict[str, object]]:
    """Deterministically fit one toy R0/R1 without consulting test data."""

    validate_toy_risk_dataset_publication(train_dataset)
    validate_toy_risk_dataset_publication(validation_dataset)
    if train_dataset.split != "train":
        raise RiskDataContractError("toy training dataset split must be 'train'")
    if validation_dataset.split != "val":
        raise RiskDataContractError("toy model-selection dataset split must be 'val'")
    if train_dataset.grid != validation_dataset.grid:
        raise RiskDataContractError("train/validation toy grids must match")
    assert_toy_split_isolation((train_dataset, validation_dataset))
    if not isinstance(optimization_steps, int) or optimization_steps < 1:
        raise ValueError("optimization_steps must be a positive integer")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if lambda_collision < 0.0:
        raise ValueError("lambda_collision must be nonnegative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    from src.datasets.risk_dataloader import collate_risk_samples

    train_batch = collate_risk_samples(
        train_dataset.samples,
        grid=train_dataset.grid,
        dataset_manifest=train_dataset.manifest,
        expected_split="train",
    )
    validation_batch = collate_risk_samples(
        validation_dataset.samples,
        grid=validation_dataset.grid,
        dataset_manifest=validation_dataset.manifest,
        expected_split="val",
    )
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = RiskModel(variant=variant, hidden_channels=hidden_channels)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=0.0
    )

    model.eval()
    with torch.no_grad():
        _, initial_losses = _batch_loss(
            model, train_batch, lambda_collision=lambda_collision
        )
        _, initial_validation_losses = _batch_loss(
            model, validation_batch, lambda_collision=lambda_collision
        )
    initial_loss = float(initial_losses["total"].item())
    history = [initial_loss]
    validation_history = [float(initial_validation_losses["total"].item())]
    best_validation_loss = validation_history[0]
    best_validation_step = 0
    best_state = copy.deepcopy(model.state_dict())
    for step in range(1, optimization_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, losses = _batch_loss(
            model, train_batch, lambda_collision=lambda_collision
        )
        losses["total"].backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            _, current_losses = _batch_loss(
                model, train_batch, lambda_collision=lambda_collision
            )
            _, current_validation_losses = _batch_loss(
                model, validation_batch, lambda_collision=lambda_collision
            )
        history.append(float(current_losses["total"].item()))
        current_validation_loss = float(current_validation_losses["total"].item())
        validation_history.append(current_validation_loss)
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_validation_step = step
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        train_output, final_losses = _batch_loss(
            model, train_batch, lambda_collision=lambda_collision
        )
        validation_output, validation_losses = _batch_loss(
            model, validation_batch, lambda_collision=lambda_collision
        )
    crossings = (
        train_output["quantiles"][:, 1:] < train_output["quantiles"][:, :-1]
    )
    predicted_collision = (train_output["p_collision"] >= 0.5).to(torch.float32)
    collision_accuracy = torch.mean(
        (predicted_collision == train_batch.targets["collision_label"]).to(
            torch.float32
        )
    )
    trajectory_sensitivity = trajectory_ablation_sensitivity(
        model, validation_batch, split="val"
    )
    metrics: dict[str, object] = {
        "variant": variant,
        "seed": seed,
        "training_split": "train",
        "selection_split": "val",
        "test_samples_used_for_training_or_selection": 0,
        "train_sample_count": len(train_dataset.samples),
        "validation_sample_count": len(validation_dataset.samples),
        "optimization_steps": optimization_steps,
        "learning_rate": float(learning_rate),
        "lambda_collision": float(lambda_collision),
        "initial_train_loss": initial_loss,
        "final_train_loss": float(final_losses["total"].item()),
        "validation_loss": float(validation_losses["total"].item()),
        "best_validation_loss": best_validation_loss,
        "best_validation_step": best_validation_step,
        "train_collision_accuracy": float(collision_accuracy.item()),
        "quantile_crossing_rate": float(crossings.to(torch.float32).mean().item()),
        "trajectory_ablation_sensitivity": trajectory_sensitivity,
        "loss_history": history,
        "validation_loss_history": validation_history,
        "train_prediction_digest_sha256": hashlib.sha256(
            train_output["quantiles"].detach().cpu().contiguous().numpy().tobytes()
            + train_output["collision_logits"]
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        ).hexdigest(),
        "validation_prediction_digest_sha256": hashlib.sha256(
            validation_output["quantiles"]
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
            + validation_output["collision_logits"]
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        ).hexdigest(),
    }
    return model, metrics


__all__ = [
    "RISK_CHECKPOINT_LAYOUT_VERSION",
    "RISK_MODEL_VARIANTS",
    "TRAJECTORY_SENSITIVITY_EPSILON",
    "RiskModel",
    "load_risk_checkpoint",
    "noncrossing_quantiles",
    "save_risk_checkpoint",
    "trajectory_ablation_sensitivity",
    "train_toy_risk_model",
]
