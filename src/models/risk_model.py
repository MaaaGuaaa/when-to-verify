"""SOP09 R0/R1/R2 risk models and checkpoint v2."""

from __future__ import annotations

import copy
import hashlib
import hmac
import io
import json
import math
from pathlib import Path
import pickle
from typing import BinaryIO, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from src.contracts import (
    LONG40_FUTURE_HORIZON_S,
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
RISK_MODEL_VARIANTS: tuple[str, ...] = ("r0", "r1", "r2")
TRAJECTORY_SENSITIVITY_EPSILON = 1e-8
R2_SCENE_INPUT_CHANNELS = 8 * N_HISTORY_CHANNELS + N_STATE_CHANNELS
R2_STEM_CHANNELS = 32
R2_D_MODEL = 128
R2_NUM_HEADS = 4
R2_DECODER_LAYERS = 2
R2_DIM_FEEDFORWARD = 256
R2_DROPOUT = 0.1
R2_MAX_SCENE_TOKENS = 400
R2_SCENE_TOKEN_GRID = (20, 20)
R2_QUERY_BINS = 8
R2_FUSION_MODES = frozenset({"cross_attention", "concat"})
R2_OCCUPANCY_AUX_CHANNELS = 32
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
        "training_stage",
        "training_subset_digest_sha256",
        "validation_risk_dataset_manifest_digest",
        "risk_dataset_family_digest",
        "global_cross_split_leakage",
        "code_commit",
        "runtime_environment_digest_sha256",
        "training_data_scale",
        "scientific_claim_eligible",
        "selected_sample_count",
        "consumed_sample_count",
        "consumed_sample_ids_digest_sha256",
    }
)
RISK_AUXILIARY_PROVENANCE_KEYS = frozenset(
    {
        "occupancy_auxiliary_enabled",
        "occupancy_sidecar_collection_digest_sha256",
        "occupancy_global_positive_count",
        "occupancy_global_negative_count",
        "occupancy_global_pos_weight",
        "occupancy_future_steps",
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


class _DeterministicGridPool(nn.Module):
    """Pool regular feature grids without CUDA adaptive-pool atomics."""

    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("grid-pool features must have shape [B,C,H,W]")
        height, width = features.shape[-2:]
        output_height, output_width = self.output_size
        if (height, width) == self.output_size:
            return features
        if (
            height >= output_height
            and width >= output_width
            and height % output_height == 0
            and width % output_width == 0
        ):
            return F.avg_pool2d(
                features,
                kernel_size=(height // output_height, width // output_width),
                stride=(height // output_height, width // output_width),
            )
        return F.interpolate(features, size=self.output_size, mode="nearest")


def _resize_grid(
    features: torch.Tensor, output_size: tuple[int, int]
) -> torch.Tensor:
    """Resize masks/features with deterministic pooling when the grid divides."""

    return _DeterministicGridPool(output_size)(features)


class _SceneContext(nn.Module):
    """Scene-only bottleneck context with both local and global receptive fields."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        context_channels = max(32, channels // 2)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, context_channels, kernel_size=1),
            nn.GELU(),
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(
                context_channels,
                context_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.GELU(),
            nn.Conv2d(
                context_channels,
                context_channels,
                kernel_size=3,
                padding=4,
                dilation=4,
            ),
            nn.GELU(),
        )
        self.expand = nn.Conv2d(context_channels, channels, kernel_size=1)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        local = self.expand(self.dilated(self.reduce(features)))
        return features + local + self.global_context(features)


class _SceneOnlyOccupancyDecoder(nn.Module):
    """Multiscale scene-only decoder for future hidden occupancy supervision."""

    def __init__(self, *, d_model: int, future_steps: int) -> None:
        super().__init__()
        self.decode_40 = self._block(d_model + 96, 96)
        self.decode_80 = self._block(96 + 64, 64)
        self.decode_full = self._block(64 + R2_STEM_CHANNELS, R2_STEM_CHANNELS)
        self.output = nn.Conv2d(
            R2_STEM_CHANNELS, future_steps, kernel_size=1
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(
        self,
        *,
        scene_stem: torch.Tensor,
        scene_80: torch.Tensor,
        scene_40: torch.Tensor,
        scene_context: torch.Tensor,
    ) -> torch.Tensor:
        decoded_40 = self.decode_40(
            torch.cat(
                (
                    F.interpolate(
                        scene_context,
                        size=scene_40.shape[-2:],
                        mode="nearest",
                    ),
                    scene_40,
                ),
                dim=1,
            )
        )
        decoded_80 = self.decode_80(
            torch.cat(
                (
                    F.interpolate(
                        decoded_40, size=scene_80.shape[-2:], mode="nearest"
                    ),
                    scene_80,
                ),
                dim=1,
            )
        )
        decoded_full = self.decode_full(
            torch.cat(
                (
                    F.interpolate(
                        decoded_80, size=scene_stem.shape[-2:], mode="nearest"
                    ),
                    scene_stem,
                ),
                dim=1,
            )
        )
        return self.output(decoded_full)


class TrajectoryQueryTransformer(nn.Module):
    """Lightweight R2 decoder that cross-attends legal trajectory queries to BEV.

    The scene path receives only flattened observed history plus current state;
    the trajectory map becomes a small set of decoder queries.  Scene memory is
    formed by CNN features with convolutional and pooled global context, without
    global scene-token self-attention.
    """

    def __init__(
        self,
        *,
        d_model: int = R2_D_MODEL,
        nhead: int = R2_NUM_HEADS,
        num_decoder_layers: int = R2_DECODER_LAYERS,
        dim_feedforward: int = R2_DIM_FEEDFORWARD,
        dropout: float = R2_DROPOUT,
        query_bins: int = R2_QUERY_BINS,
        fusion_mode: str = "cross_attention",
        occupancy_aux_enabled: bool = False,
        occupancy_future_steps: int = R2_OCCUPANCY_AUX_CHANNELS,
    ) -> None:
        super().__init__()
        for name, value in (
            ("d_model", d_model),
            ("nhead", nhead),
            ("num_decoder_layers", num_decoder_layers),
            ("dim_feedforward", dim_feedforward),
            ("query_bins", query_bins),
            ("occupancy_future_steps", occupancy_future_steps),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"R2 {name} must be a positive integer")
        if d_model % nhead != 0:
            raise ValueError("R2 d_model must be divisible by nhead")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("R2 dropout must lie in [0,1)")
        if fusion_mode not in R2_FUSION_MODES:
            raise ValueError(f"R2 fusion_mode must be one of {sorted(R2_FUSION_MODES)}")
        if not isinstance(occupancy_aux_enabled, bool):
            raise ValueError("occupancy_aux_enabled must be boolean")

        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_decoder_layers = int(num_decoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.query_bins = int(query_bins)
        self.fusion_mode = fusion_mode
        self.occupancy_aux_enabled = occupancy_aux_enabled
        self.occupancy_future_steps = int(occupancy_future_steps)
        self.max_scene_tokens = R2_MAX_SCENE_TOKENS
        self.scene_token_grid = R2_SCENE_TOKEN_GRID

        # This branch has no trajectory channels by construction: 16 flattened
        # history channels plus 9 current-state channels is the frozen total 25.
        self.scene_stem = nn.Sequential(
            nn.Conv2d(R2_SCENE_INPUT_CHANNELS, R2_STEM_CHANNELS, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(R2_STEM_CHANNELS, R2_STEM_CHANNELS, 3, padding=1),
            nn.GELU(),
        )
        self.scene_down_80 = nn.Sequential(
            nn.Conv2d(R2_STEM_CHANNELS, 64, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.scene_down_40 = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.scene_down_20 = nn.Sequential(
            nn.Conv2d(96, d_model, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.scene_context = _SceneContext(d_model)
        self.trajectory_encoder = nn.Sequential(
            nn.Conv2d(N_TRAJECTORY_CHANNELS, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, d_model, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.scene_token_pool = _DeterministicGridPool(R2_SCENE_TOKEN_GRID)
        self.scene_position = nn.Linear(2, d_model, bias=False)
        self.trajectory_position = nn.Linear(2, d_model, bias=False)
        self.time_embedding = nn.Embedding(query_bins, d_model)
        self.scene_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.trajectory_decoder: nn.TransformerDecoder | None = None
        self.trajectory_concat_encoder: nn.TransformerEncoder | None = None
        if fusion_mode == "cross_attention":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.trajectory_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_decoder_layers,
                norm=nn.LayerNorm(d_model),
            )
        else:
            query_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.trajectory_concat_encoder = nn.TransformerEncoder(
                query_encoder_layer,
                num_layers=num_decoder_layers,
                norm=nn.LayerNorm(d_model),
            )
        fusion_features = (
            d_model + ROBOT_STATE_DIM
            if fusion_mode == "cross_attention"
            else 2 * d_model + ROBOT_STATE_DIM
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.quantile_head = nn.Linear(d_model, len(QUANTILE_LEVELS))
        self.collision_head = nn.Linear(d_model, 1)
        self.occupancy_aux_decoder: nn.Module | None
        if occupancy_aux_enabled:
            self.occupancy_aux_decoder = _SceneOnlyOccupancyDecoder(
                d_model=d_model,
                future_steps=occupancy_future_steps,
            )
        else:
            self.occupancy_aux_decoder = None

    def export_config(self) -> dict[str, object]:
        return {
            "variant": "r2",
            "history_steps": 8,
            "r2_d_model": self.d_model,
            "r2_nhead": self.nhead,
            "r2_num_decoder_layers": self.num_decoder_layers,
            "r2_dim_feedforward": self.dim_feedforward,
            "r2_dropout": self.dropout,
            "r2_query_bins": self.query_bins,
            "r2_fusion_mode": self.fusion_mode,
            "occupancy_aux_enabled": self.occupancy_aux_enabled,
            "occupancy_future_steps": self.occupancy_future_steps,
        }

    @staticmethod
    def _position_tokens(
        features: torch.Tensor, projection: nn.Linear
    ) -> torch.Tensor:
        """Create lightweight 2-D positions without adding input channels."""

        height, width = features.shape[-2:]
        y = torch.linspace(
            -1.0, 1.0, height, device=features.device, dtype=features.dtype
        )
        x = torch.linspace(
            -1.0, 1.0, width, device=features.device, dtype=features.dtype
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=-1).reshape(1, height * width, 2)
        return projection(coordinates).expand(features.shape[0], -1, -1)

    def _scene_tokens(self, scene_features: torch.Tensor) -> torch.Tensor:
        height, width = scene_features.shape[-2:]
        if height * width > self.max_scene_tokens:
            scene_features = self.scene_token_pool(scene_features)
        tokens = scene_features.flatten(2).transpose(1, 2)
        return self.scene_norm(
            tokens + self._position_tokens(scene_features, self.scene_position)
        )

    def _trajectory_queries(
        self, trajectory_features: torch.Tensor, trajectory_channels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool legal swept cells into ordered time-to-arrival query tokens."""

        target_size = trajectory_features.shape[-2:]
        swept = _resize_grid(trajectory_channels[:, :1], target_size).clamp(0.0, 1.0)
        arrival_seconds = _resize_grid(
            trajectory_channels[:, 1:2] * trajectory_channels[:, :1], target_size
        ) / swept.clamp_min(torch.finfo(trajectory_features.dtype).eps)
        arrival = arrival_seconds / LONG40_FUTURE_HORIZON_S
        weights: list[torch.Tensor] = []
        for index in range(self.query_bins):
            lower = float(index) / self.query_bins
            upper = float(index + 1) / self.query_bins
            if index + 1 == self.query_bins:
                in_bin = (arrival >= lower) & (arrival <= upper)
            else:
                in_bin = (arrival >= lower) & (arrival < upper)
            weights.append(swept * in_bin.to(dtype=swept.dtype))
        query_weights = torch.cat(weights, dim=1)
        weight_sums = query_weights.sum(dim=(-2, -1))
        valid = weight_sums > 0.0
        no_valid = ~valid.any(dim=1)
        if torch.any(no_valid):
            fallback = swept[:, 0]
            fallback_is_empty = fallback.sum(dim=(-2, -1)) <= 0.0
            fallback = torch.where(
                fallback_is_empty[:, None, None], torch.ones_like(fallback), fallback
            )
            query_weights = query_weights.clone()
            query_weights[no_valid, 0] = fallback[no_valid]
            weight_sums = query_weights.sum(dim=(-2, -1))
            valid = weight_sums > 0.0
        normalized = query_weights / weight_sums.clamp_min(
            torch.finfo(trajectory_features.dtype).eps
        )[:, :, None, None]
        query_features = torch.einsum(
            "bqhw,bdhw->bqd", normalized, trajectory_features
        )
        height, width = target_size
        y = torch.linspace(
            -1.0, 1.0, height, device=trajectory_features.device, dtype=trajectory_features.dtype
        )
        x = torch.linspace(
            -1.0, 1.0, width, device=trajectory_features.device, dtype=trajectory_features.dtype
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=-1)
        query_coordinates = torch.einsum(
            "bqhw,hwc->bqc", normalized, coordinates
        )
        time_ids = torch.arange(
            self.query_bins, device=trajectory_features.device, dtype=torch.long
        )
        query_tokens = (
            query_features
            + self.trajectory_position(query_coordinates)
            + self.time_embedding(time_ids).unsqueeze(0)
        )
        return self.query_norm(query_tokens), valid

    @staticmethod
    def _masked_max(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        masked = tokens.masked_fill(~valid[:, :, None], float("-inf"))
        return masked.amax(dim=1)

    def forward(
        self,
        scene_inputs: torch.Tensor,
        trajectory_channels: torch.Tensor,
        robot_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            scene_inputs.ndim != 4
            or scene_inputs.shape[1] != R2_SCENE_INPUT_CHANNELS
        ):
            raise RiskDataContractError("R2 scene inputs must have shape [B,25,H,W]")
        if (
            trajectory_channels.ndim != 4
            or trajectory_channels.shape[1] != N_TRAJECTORY_CHANNELS
        ):
            raise RiskDataContractError(
                "R2 trajectory inputs must have shape [B,4,H,W]"
            )
        if scene_inputs.shape[0] != trajectory_channels.shape[0] or (
            scene_inputs.shape[-2:] != trajectory_channels.shape[-2:]
        ):
            raise RiskDataContractError("R2 scene and trajectory inputs must align")
        if robot_state.shape != (scene_inputs.shape[0], ROBOT_STATE_DIM):
            raise RiskDataContractError("R2 robot_state must have shape [B,2]")

        scene_stem = self.scene_stem(scene_inputs)
        scene_80 = self.scene_down_80(scene_stem)
        scene_40 = self.scene_down_40(scene_80)
        scene_context = self.scene_context(self.scene_down_20(scene_40))
        scene_tokens = self._scene_tokens(scene_context)
        trajectory_tokens, valid_queries = self._trajectory_queries(
            self.trajectory_encoder(trajectory_channels), trajectory_channels
        )
        padding_mask = ~valid_queries
        if self.fusion_mode == "cross_attention":
            assert self.trajectory_decoder is not None
            decoded_queries = self.trajectory_decoder(
                trajectory_tokens,
                scene_tokens,
                tgt_key_padding_mask=padding_mask,
            )
            fusion_inputs = torch.cat(
                (self._masked_max(decoded_queries, valid_queries), robot_state), dim=1
            )
        else:
            assert self.trajectory_concat_encoder is not None
            decoded_queries = self.trajectory_concat_encoder(
                trajectory_tokens, src_key_padding_mask=padding_mask
            )
            fusion_inputs = torch.cat(
                (
                    self._masked_max(decoded_queries, valid_queries),
                    scene_tokens.mean(dim=1),
                    robot_state,
                ),
                dim=1,
            )
        fused = self.fusion(
            fusion_inputs
        )
        collision_logits = self.collision_head(fused).squeeze(-1)
        output = {
            "quantiles": noncrossing_quantiles(self.quantile_head(fused)),
            "collision_logits": collision_logits,
            "p_collision": torch.sigmoid(collision_logits),
        }
        if self.occupancy_aux_decoder is not None:
            output["occupancy_aux_logits"] = self.occupancy_aux_decoder(
                scene_stem=scene_stem,
                scene_80=scene_80,
                scene_40=scene_40,
                scene_context=scene_context,
            )
        return output


class RiskModel(nn.Module):
    """R0/R1 baselines plus the R2 trajectory-query Transformer."""

    def __init__(
        self,
        *,
        variant: str = "r0",
        hidden_channels: int | None = None,
        history_steps: int = 8,
        r2_d_model: int = R2_D_MODEL,
        r2_nhead: int = R2_NUM_HEADS,
        r2_num_decoder_layers: int = R2_DECODER_LAYERS,
        r2_dim_feedforward: int = R2_DIM_FEEDFORWARD,
        r2_dropout: float = R2_DROPOUT,
        r2_query_bins: int = R2_QUERY_BINS,
        r2_fusion_mode: str = "cross_attention",
        occupancy_aux_enabled: bool = False,
        occupancy_future_steps: int = R2_OCCUPANCY_AUX_CHANNELS,
        # Read legacy pre-refactor R2 checkpoints without emitting this shape
        # back into new checkpoint configs.
        d_model: int | None = None,
        nhead: int | None = None,
        num_decoder_layers: int | None = None,
        dim_feedforward: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        if variant not in RISK_MODEL_VARIANTS:
            raise ValueError(f"variant must be one of {RISK_MODEL_VARIANTS}")
        if not isinstance(occupancy_aux_enabled, bool):
            raise ValueError("occupancy_aux_enabled must be boolean")
        if variant != "r2" and occupancy_aux_enabled:
            raise ValueError("occupancy_aux_enabled is available only for r2")
        if history_steps != 8:
            raise ValueError("SOP09 frozen history_steps must equal 8")
        self.variant = variant
        self.history_steps = int(history_steps)
        if variant == "r2":
            if hidden_channels is not None:
                raise ValueError("hidden_channels is unavailable for the r2 model variant")
            legacy_values = {
                "d_model": d_model,
                "nhead": nhead,
                "num_decoder_layers": num_decoder_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
            }
            current_values = {
                "d_model": r2_d_model,
                "nhead": r2_nhead,
                "num_decoder_layers": r2_num_decoder_layers,
                "dim_feedforward": r2_dim_feedforward,
                "dropout": r2_dropout,
            }
            for name, legacy_value in legacy_values.items():
                if legacy_value is not None:
                    current_values[name] = legacy_value
            self.hidden_channels = None
            self.r2_model = TrajectoryQueryTransformer(
                d_model=int(current_values["d_model"]),
                nhead=int(current_values["nhead"]),
                num_decoder_layers=int(current_values["num_decoder_layers"]),
                dim_feedforward=int(current_values["dim_feedforward"]),
                dropout=float(current_values["dropout"]),
                query_bins=r2_query_bins,
                fusion_mode=r2_fusion_mode,
                occupancy_aux_enabled=occupancy_aux_enabled,
                occupancy_future_steps=occupancy_future_steps,
            )
            return
        if hidden_channels is None:
            hidden_channels = 16
        if type(hidden_channels) is not int or hidden_channels < 1:
            raise ValueError("hidden_channels must be a positive integer for r0/r1")
        self.hidden_channels = int(hidden_channels)
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
        if self.variant == "r2":
            return self.r2_model.export_config()
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
        elif self.variant == "r1":
            hidden: torch.Tensor | None = None
            for step in range(steps):
                encoded = self.history_encoder(history[:, step])
                hidden = self.temporal_cell(encoded, hidden)
            assert hidden is not None
            temporal_features = self.pool(hidden).flatten(1)
            context = self.context_encoder(torch.cat((state, trajectory), dim=1))
            context_features = self.pool(context).flatten(1)
            features = torch.cat((temporal_features, context_features), dim=1)
        else:
            scene_inputs = torch.cat(
                (history.reshape(batch, steps * channels, height, width), state),
                dim=1,
            )
            return self.r2_model(scene_inputs, trajectory, robot_state)
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
    actual_keys = set(provenance)
    permits_auxiliary = (
        mode == "production"
        and actual_keys
        == (RISK_PRODUCTION_PROVENANCE_KEYS | RISK_AUXILIARY_PROVENANCE_KEYS)
    )
    if actual_keys != expected_keys and not permits_auxiliary:
        expected_with_auxiliary = (
            expected_keys | RISK_AUXILIARY_PROVENANCE_KEYS
            if mode == "production"
            else expected_keys
        )
        missing = sorted(expected_with_auxiliary - actual_keys)
        unexpected = sorted(repr(key) for key in actual_keys - expected_with_auxiliary)
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
    nullable_fields = (
        {
            "validation_risk_dataset_manifest_digest",
            "risk_dataset_family_digest",
        }
        if mode == "production"
        else set()
    )
    string_fields = expected_keys - {
        "schema_version",
        "channel_spec",
        "model_variant",
        "seed",
        "scientific_claim_eligible",
        "selected_sample_count",
        "consumed_sample_count",
    } - nullable_fields
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
        return

    if permits_auxiliary:
        if provenance["occupancy_auxiliary_enabled"] is not True:
            raise RiskDataContractError(
                "auxiliary checkpoint provenance must record occupancy_auxiliary_enabled=true"
            )
        sidecar_digest = provenance["occupancy_sidecar_collection_digest_sha256"]
        if (
            not isinstance(sidecar_digest, str)
            or len(sidecar_digest) != 64
            or any(character not in "0123456789abcdef" for character in sidecar_digest)
        ):
            raise RiskDataContractError(
                "checkpoint occupancy sidecar collection digest must be a lowercase SHA-256 digest"
            )
        positive_count = provenance["occupancy_global_positive_count"]
        negative_count = provenance["occupancy_global_negative_count"]
        if (
            type(positive_count) is not int
            or type(negative_count) is not int
            or positive_count < 1
            or negative_count < 1
        ):
            raise RiskDataContractError(
                "checkpoint occupancy class counts must be positive integers"
            )
        pos_weight = provenance["occupancy_global_pos_weight"]
        expected_weight = float(negative_count) / float(positive_count)
        if (
            type(pos_weight) not in {int, float}
            or not math.isfinite(float(pos_weight))
            or float(pos_weight) <= 0.0
            or not math.isclose(float(pos_weight), expected_weight, rel_tol=0.0, abs_tol=0.0)
        ):
            raise RiskDataContractError(
                "checkpoint occupancy_global_pos_weight must equal negative/positive count"
            )
        if provenance["occupancy_future_steps"] != 32:
            raise RiskDataContractError(
                "checkpoint auxiliary occupancy horizon must equal 32"
            )

    digest_contract = {
        "g1_split_manifest_digest": (32, "BLAKE2b-128"),
        "target_type_policy_digest": (32, "BLAKE2b-128"),
        "risk_dataset_manifest_digest": (64, "SHA-256"),
        "dynamic_objects_config_digest": (64, "SHA-256"),
        "config_digest": (64, "SHA-256"),
        "training_subset_digest_sha256": (64, "SHA-256"),
        "runtime_environment_digest_sha256": (64, "SHA-256"),
        "consumed_sample_ids_digest_sha256": (64, "SHA-256"),
    }
    for field, (length, algorithm) in digest_contract.items():
        value = provenance[field]
        if len(value) != length or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise RiskDataContractError(
                f"checkpoint provenance {field} must be a lowercase {algorithm} digest"
            )
    code_commit = provenance["code_commit"]
    if (
        not isinstance(code_commit, str)
        or len(code_commit) != 40
        or any(character not in "0123456789abcdef" for character in code_commit)
    ):
        raise RiskDataContractError(
            "checkpoint provenance code_commit must be a lowercase 40-character commit"
        )
    for field in ("selected_sample_count", "consumed_sample_count"):
        value = provenance[field]
        if type(value) is not int or value < 1:
            raise RiskDataContractError(
                f"checkpoint provenance {field} must be a positive integer"
            )
    if provenance["consumed_sample_count"] > provenance["selected_sample_count"]:
        raise RiskDataContractError(
            "checkpoint provenance consumed_sample_count exceeds selected_sample_count"
        )
    data_scale = provenance["training_data_scale"]
    eligibility = provenance["scientific_claim_eligible"]
    if data_scale not in {
        "one_shard_smoke",
        "fixture_standin",
        "real_1k",
        "formal_50k",
    }:
        raise RiskDataContractError(
            "checkpoint provenance training_data_scale mismatch"
        )
    if type(eligibility) is not bool:
        raise RiskDataContractError(
            "checkpoint provenance scientific_claim_eligible must be boolean"
        )
    if (data_scale in {"one_shard_smoke", "fixture_standin"}) != (
        eligibility is False
    ):
        raise RiskDataContractError(
            "checkpoint provenance scientific claim eligibility/scale mismatch"
        )
    if data_scale == "real_1k" and (
        provenance["selected_sample_count"] != 1000
        or provenance["consumed_sample_count"] != 1000
    ):
        raise RiskDataContractError(
            "real_1k checkpoint provenance requires exactly 1000 consumed samples"
        )
    training_stage = provenance["training_stage"]
    if training_stage not in {
        "one_shard_smoke",
        "real_1k_overfit",
        "formal_50k",
    }:
        raise RiskDataContractError("checkpoint provenance training_stage mismatch")
    if training_stage == "one_shard_smoke" and data_scale != "one_shard_smoke":
        raise RiskDataContractError(
            "one_shard_smoke checkpoint provenance scale mismatch"
        )
    if training_stage == "real_1k_overfit":
        if data_scale not in {"fixture_standin", "real_1k"}:
            raise RiskDataContractError(
                "real_1k_overfit checkpoint provenance scale mismatch"
            )
        if provenance["consumed_sample_count"] != provenance["selected_sample_count"]:
            raise RiskDataContractError(
                "real_1k_overfit checkpoint must consume all selected samples"
            )
        if data_scale == "fixture_standin" and provenance[
            "selected_sample_count"
        ] >= 1000:
            raise RiskDataContractError(
                "fixture_standin checkpoint must contain fewer than 1000 samples"
            )
    if training_stage == "formal_50k":
        if data_scale not in {"fixture_standin", "formal_50k"}:
            raise RiskDataContractError(
                "formal_50k checkpoint provenance scale mismatch"
            )
        if provenance["consumed_sample_count"] != provenance[
            "selected_sample_count"
        ]:
            raise RiskDataContractError(
                "formal_50k checkpoint must consume all selected samples"
            )
        if data_scale == "formal_50k" and provenance[
            "selected_sample_count"
        ] != 50_000:
            raise RiskDataContractError(
                "formal_50k checkpoint provenance requires exactly 50000 samples"
            )
        if data_scale == "fixture_standin" and provenance[
            "selected_sample_count"
        ] == 50_000:
            raise RiskDataContractError(
                "formal checkpoint scale must match its exact 50000-sample identity"
            )
    if provenance["global_cross_split_leakage"] not in {
        "NOT_PROVEN",
        "PROVEN",
    }:
        raise RiskDataContractError(
            "checkpoint provenance global_cross_split_leakage must be NOT_PROVEN or PROVEN"
        )
    validation_digest = provenance["validation_risk_dataset_manifest_digest"]
    family_digest = provenance["risk_dataset_family_digest"]
    for field, value in (
        ("validation_risk_dataset_manifest_digest", validation_digest),
        ("risk_dataset_family_digest", family_digest),
    ):
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RiskDataContractError(
                f"checkpoint provenance {field} must be None or a lowercase SHA-256 digest"
            )
    if training_stage == "formal_50k":
        if validation_digest is None or family_digest is None:
            raise RiskDataContractError(
                "formal_50k checkpoint requires validation and dataset-family digests"
            )
        if provenance["global_cross_split_leakage"] != "PROVEN":
            raise RiskDataContractError(
                "formal_50k checkpoint requires global_cross_split_leakage=PROVEN"
            )
    elif (
        validation_digest is not None
        or family_digest is not None
        or provenance["global_cross_split_leakage"] != "NOT_PROVEN"
    ):
        raise RiskDataContractError(
            "smoke/1k checkpoint provenance must use no validation/family digest and NOT_PROVEN"
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
    if mode == "production":
        model_config = model.export_config()
        model_uses_auxiliary = (
            model.variant == "r2"
            and model_config.get("occupancy_aux_enabled") is True
        )
        provenance_binds_auxiliary = (
            provenance.get("occupancy_auxiliary_enabled") is True
        )
        if model_uses_auxiliary != provenance_binds_auxiliary:
            raise RiskDataContractError(
                "production auxiliary provenance does not match the model"
            )
        if model_uses_auxiliary and (
            model_config.get("occupancy_future_steps")
            != provenance.get("occupancy_future_steps")
        ):
            raise RiskDataContractError(
                "production auxiliary model/provenance occupancy horizon mismatch"
            )
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
    path: str | Path | bytes | bytearray | memoryview | BinaryIO,
    *,
    expected_mode: str,
    expected_provenance: Mapping[str, object] | None = None,
) -> tuple[RiskModel, dict[str, object]]:
    """Load a checkpoint only after version/mode/provenance validation."""

    source: object
    if isinstance(path, (bytes, bytearray, memoryview)):
        source = io.BytesIO(bytes(path))
    elif isinstance(path, (str, Path)):
        source = Path(path)
    elif hasattr(path, "read") and hasattr(path, "seek"):
        source = path
        try:
            path.seek(0)
        except (OSError, ValueError) as error:
            raise RiskDataContractError(
                f"unable to seek risk checkpoint snapshot: {error}"
            ) from error
    else:
        raise TypeError("checkpoint source must be a path, bytes, or seekable file")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
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


def compute_risk_batch_loss(
    model: RiskModel,
    batch: object,
    *,
    lambda_collision: float,
    lambda_occupancy_aux: float = 0.0,
    occupancy_target: torch.Tensor | None = None,
    occupancy_pos_weight: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Evaluate the risk loss without admitting auxiliary labels as inputs.

    An authenticated sidecar provides ``occupancy_targets.hidden_risk_occupancy``
    on a separate batch namespace.  Model inputs remain unchanged, and enabling
    an auxiliary head without that label fails closed.
    """

    output = model(batch.model_inputs)
    if occupancy_target is None:
        occupancy_targets = getattr(batch, "occupancy_targets", None)
        if isinstance(occupancy_targets, Mapping):
            candidate = occupancy_targets.get("hidden_risk_occupancy")
            if isinstance(candidate, torch.Tensor):
                occupancy_target = candidate
    if output.get("occupancy_aux_logits") is not None and occupancy_target is None:
        raise RiskDataContractError(
            "occupancy auxiliary output requires a separate 32-step occupancy target"
        )
    losses = risk_loss(
        output,
        risk_severity=batch.targets["risk_severity"],
        collision_label=batch.targets["collision_label"],
        lambda_collision=lambda_collision,
        occupancy_target=occupancy_target,
        lambda_occupancy_aux=lambda_occupancy_aux,
        occupancy_pos_weight=occupancy_pos_weight,
    )
    return output, losses


def production_trajectory_query_sensitivity(
    model: RiskModel,
    batch: object,
) -> dict[str, object]:
    """Measure label-free conditioning on authenticated production queries.

    A complete trajectory query consists of its spatial trajectory channels and
    matching ``(v, omega)`` robot state, so both components are always permuted
    together.  Targets are deliberately never inspected.
    """

    if getattr(batch, "split", None) != "train":
        raise RiskDataContractError(
            "production trajectory sensitivity requires a train batch"
        )
    model_inputs = getattr(batch, "model_inputs", None)
    if not isinstance(model_inputs, Mapping):
        raise RiskDataContractError(
            "production trajectory sensitivity requires model_inputs"
        )
    provenance = getattr(batch, "provenance", None)
    if not isinstance(provenance, Mapping) or provenance.get("mode") != "production":
        raise RiskDataContractError(
            "production trajectory sensitivity requires authenticated production provenance"
        )
    sample_ids = tuple(getattr(batch, "sample_ids", ()))
    trajectory = model_inputs.get("trajectory_channels")
    robot_state = model_inputs.get("robot_state")
    if not isinstance(trajectory, torch.Tensor) or not isinstance(
        robot_state, torch.Tensor
    ):
        raise RiskDataContractError(
            "production trajectory sensitivity requires tensor query components"
        )
    sample_count = len(sample_ids)
    if (
        sample_count < 2
        or trajectory.shape[0] != sample_count
        or robot_state.shape[0] != sample_count
    ):
        raise RiskDataContractError(
            "production trajectory sensitivity requires at least two aligned rows"
        )

    indices = torch.arange(sample_count, device=trajectory.device)
    best_permutation: torch.Tensor | None = None
    best_changed_count = -1
    best_shift = -1
    for shift in range(1, sample_count):
        permutation = torch.roll(indices, shifts=-shift)
        trajectory_changed = torch.any(
            (trajectory[permutation] != trajectory).reshape(sample_count, -1), dim=1
        )
        robot_state_changed = torch.any(
            (robot_state[permutation] != robot_state).reshape(sample_count, -1), dim=1
        )
        changed_count = int(
            torch.count_nonzero(trajectory_changed | robot_state_changed).item()
        )
        if changed_count > best_changed_count:
            best_permutation = permutation
            best_changed_count = changed_count
            best_shift = shift
    if best_permutation is None or best_changed_count < 1:
        raise RiskDataContractError(
            "production rows do not provide a changed legal trajectory query"
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
        torch.mean(torch.abs(reference["quantiles"] - counterfactual["quantiles"]))
        .detach()
        .cpu()
        .item()
    )
    logit_delta = float(
        torch.mean(
            torch.abs(
                reference["collision_logits"] - counterfactual["collision_logits"]
            )
        )
        .detach()
        .cpu()
        .item()
    )
    probability_delta = float(
        torch.mean(
            torch.abs(reference["p_collision"] - counterfactual["p_collision"])
        )
        .detach()
        .cpu()
        .item()
    )
    combined_delta = quantile_delta + logit_delta + probability_delta
    if not all(
        torch.isfinite(torch.tensor(value)).item()
        for value in (
            quantile_delta,
            logit_delta,
            probability_delta,
            combined_delta,
        )
    ):
        raise RiskDataContractError(
            "production trajectory sensitivity contains NaN/Inf"
        )
    return {
        "protocol": "deterministic_permutation_of_authenticated_production_query",
        "diagnostic_kind": "legal_counterfactual_query_conditioning_sensitivity",
        "split": "train",
        "sample_count": sample_count,
        "source_dataset_manifest_digest": provenance.get(
            "risk_dataset_manifest_digest"
        ),
        "query_components_permuted": ["trajectory_channels", "robot_state"],
        "permutation_shift": best_shift,
        "changed_query_count": best_changed_count,
        "unchanged_query_count": sample_count - best_changed_count,
        "labels_accessed": False,
        "used_for_training_or_selection": False,
        "quantile_mean_absolute_delta": quantile_delta,
        "collision_logit_mean_absolute_delta": logit_delta,
        "collision_probability_mean_absolute_delta": probability_delta,
        "combined_mean_absolute_delta": combined_delta,
        "materiality_threshold": TRAJECTORY_SENSITIVITY_EPSILON,
        "materially_sensitive": combined_delta > TRAJECTORY_SENSITIVITY_EPSILON,
    }


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
    """Deterministically fit one toy R0/R1/R2 without consulting test data."""

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
    model = RiskModel(
        variant=variant,
        hidden_channels=None if variant == "r2" else hidden_channels,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=0.0
    )

    model.eval()
    with torch.no_grad():
        _, initial_losses = compute_risk_batch_loss(
            model, train_batch, lambda_collision=lambda_collision
        )
        _, initial_validation_losses = compute_risk_batch_loss(
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
        _, losses = compute_risk_batch_loss(
            model, train_batch, lambda_collision=lambda_collision
        )
        losses["total"].backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            _, current_losses = compute_risk_batch_loss(
                model, train_batch, lambda_collision=lambda_collision
            )
            _, current_validation_losses = compute_risk_batch_loss(
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
        train_output, final_losses = compute_risk_batch_loss(
            model, train_batch, lambda_collision=lambda_collision
        )
        validation_output, validation_losses = compute_risk_batch_loss(
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
    "R2_DECODER_LAYERS",
    "R2_DIM_FEEDFORWARD",
    "R2_D_MODEL",
    "R2_DROPOUT",
    "R2_MAX_SCENE_TOKENS",
    "R2_NUM_HEADS",
    "R2_OCCUPANCY_AUX_CHANNELS",
    "R2_QUERY_BINS",
    "R2_SCENE_INPUT_CHANNELS",
    "R2_SCENE_TOKEN_GRID",
    "R2_STEM_CHANNELS",
    "RISK_CHECKPOINT_LAYOUT_VERSION",
    "RISK_MODEL_VARIANTS",
    "TRAJECTORY_SENSITIVITY_EPSILON",
    "RiskModel",
    "TrajectoryQueryTransformer",
    "compute_risk_batch_loss",
    "load_risk_checkpoint",
    "noncrossing_quantiles",
    "production_trajectory_query_sensitivity",
    "save_risk_checkpoint",
    "trajectory_ablation_sensitivity",
    "train_toy_risk_model",
]
