from dataclasses import replace

import numpy as np
import pytest

from src.contracts import VerificationSample, assert_no_oracle_leakage
from src.datasets.verification_dataset import (
    build_verification_samples,
    verification_model_inputs,
)
from tests.test_verification_dataset import _source_and_library


LEGAL_INPUT_KEYS = {
    "bev_history",
    "state_channels",
    "trajectory_channels",
    "verification_fov_mask",
    "verification_action_vector",
}
FORBIDDEN_TOKENS = {
    "oracle",
    "future",
    "world",
    "post_risk",
    "post_observation",
    "value_target",
    "br_before",
    "scenario",
}


def _assert_recursive_input_is_clean(value, *, path="inputs"):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            assert not any(token in lowered for token in FORBIDDEN_TOKENS), (
                f"forbidden model input key at {path}.{key}"
            )
            _assert_recursive_input_is_clean(child, path=f"{path}.{key}")
        return
    assert isinstance(value, np.ndarray), f"unexpected model input at {path}"
    assert value.dtype == np.float32
    assert np.isfinite(value).all()


def test_model_input_projection_is_an_exact_oracle_free_allowlist():
    grid, library, source = _source_and_library()
    sample = build_verification_samples(source, library=library, grid=grid)[0]

    inputs = verification_model_inputs(sample)

    assert set(inputs) == LEGAL_INPUT_KEYS
    _assert_recursive_input_is_clean(dict(inputs))
    assert "metadata" not in inputs
    assert "useful_target" not in inputs
    for key in LEGAL_INPUT_KEYS:
        assert inputs[key] is getattr(sample, key)
    assert_no_oracle_leakage(VerificationSample)


@pytest.mark.parametrize(
    "bad_key",
    ["oracle_world_id", "future_actor_pose", "post_observation_occupancy"],
)
def test_provenance_rejects_label_side_oracle_payload_keys(bad_key):
    grid, library, source = _source_and_library()
    provenance = dict(source.provenance)
    provenance[bad_key] = "must-not-enter-sample"

    with pytest.raises(ValueError, match="forbidden"):
        build_verification_samples(
            replace(source, provenance=provenance), library=library, grid=grid
        )
