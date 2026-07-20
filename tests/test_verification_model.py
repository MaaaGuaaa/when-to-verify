from pathlib import Path

import pytest
import torch

from src.contracts import build_grid_spec
from src.models.verification_model import (
    VERIFICATION_MODEL_VERSION,
    VerificationValueModel,
    load_verify_model_config,
)
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs/verify_model.yaml"


def _inputs(batch_size: int = 2):
    grid = build_grid_spec(load_config(ROOT / "configs/base.yaml"))
    return grid, {
        "bev_history": torch.zeros(
            batch_size,
            grid.history_steps,
            grid.n_history_channels,
            grid.height,
            grid.width,
            dtype=torch.float32,
        ),
        "state_channels": torch.zeros(
            batch_size,
            grid.n_state_channels,
            grid.height,
            grid.width,
            dtype=torch.float32,
        ),
        "trajectory_channels": torch.zeros(
            batch_size,
            grid.n_trajectory_channels,
            grid.height,
            grid.width,
            dtype=torch.float32,
        ),
        "verification_fov_mask": torch.zeros(
            batch_size, 1, grid.height, grid.width, dtype=torch.float32
        ),
        "verification_action_vector": torch.zeros(
            batch_size, 3, dtype=torch.float32
        ),
    }


def test_v0_forward_has_two_finite_batch_outputs_and_exact_input_api():
    grid, inputs = _inputs()
    config = load_verify_model_config(MODEL_CONFIG)
    model = VerificationValueModel(
        grid=grid,
        config=config.model,
        initialization_seed=config.training.seed,
    )

    prediction = model(**inputs)

    assert model.version == VERIFICATION_MODEL_VERSION
    assert prediction.g_pred.shape == (2,)
    assert prediction.useful_logit.shape == (2,)
    assert prediction.g_pred.dtype == torch.float32
    assert prediction.useful_logit.dtype == torch.float32
    assert torch.isfinite(prediction.g_pred).all()
    assert torch.isfinite(prediction.useful_logit).all()
    with pytest.raises(TypeError, match="post_action_oracle"):
        model(**inputs, post_action_oracle=torch.zeros(2, 1, 1, 1))


def test_initialization_is_deterministic_without_mutating_global_rng():
    grid, _ = _inputs(batch_size=1)
    config = load_verify_model_config(MODEL_CONFIG)
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    first = VerificationValueModel(
        grid=grid,
        config=config.model,
        initialization_seed=77,
    )
    after = torch.random.get_rng_state().clone()
    second = VerificationValueModel(
        grid=grid,
        config=config.model,
        initialization_seed=77,
    )

    assert torch.equal(before, after)
    for left, right in zip(
        first.state_dict().values(), second.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "verification_action_vector",
            torch.zeros(2, 4, dtype=torch.float32),
            "verification_action_vector",
        ),
        (
            "verification_fov_mask",
            torch.zeros(2, 2, 160, 160, dtype=torch.float32),
            "verification_fov_mask",
        ),
        (
            "state_channels",
            torch.zeros(2, 9, 160, 160, dtype=torch.float64),
            "float32",
        ),
    ],
)
def test_forward_rejects_shape_or_dtype_contract_violations(
    field, replacement, message
):
    grid, inputs = _inputs()
    config = load_verify_model_config(MODEL_CONFIG)
    model = VerificationValueModel(
        grid=grid,
        config=config.model,
        initialization_seed=config.training.seed,
    )
    inputs[field] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        model(**inputs)


def test_forward_and_loss_path_produce_finite_cpu_gradients():
    grid, inputs = _inputs()
    config = load_verify_model_config(MODEL_CONFIG)
    model = VerificationValueModel(
        grid=grid,
        config=config.model,
        initialization_seed=config.training.seed,
    )
    prediction = model(**inputs)
    scalar = prediction.g_pred.square().mean() + prediction.useful_logit.square().mean()
    scalar.backward()

    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)


def test_model_config_is_strict_and_schema_bound(tmp_path):
    config = load_verify_model_config(MODEL_CONFIG)
    assert config.schema_version == "3.0.0"
    assert config.model.version == VERIFICATION_MODEL_VERSION
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(MODEL_CONFIG.read_text() + "unexpected: true\n")

    with pytest.raises(ValueError, match="keys"):
        load_verify_model_config(invalid)
