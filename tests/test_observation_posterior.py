from pathlib import Path

import numpy as np
import pytest

from src.generation.counterfactual_verify import (
    CounterfactualObservation,
    fit_signature_normalizer,
)
from src.generation.observation_posterior import (
    exact_observation_posterior,
    load_observation_posterior_config,
    observable_observation_digest,
    soft_observation_posterior,
    validate_posterior_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "verification_gt.yaml"


def test_exact_posterior_matches_hand_grouping():
    digests = ("a", "a", "b", "c", "c")
    posterior = exact_observation_posterior(digests)
    expected = np.asarray(
        [
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(posterior, expected)
    validate_posterior_matrix(posterior, size=5)


def test_soft_posterior_is_finite_normalized_and_temperature_sensitive():
    train = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 0, 1],
            [3, 3, 3, 3, 3, 1, 0],
        ],
        dtype=np.float32,
    )
    normalizer = fit_signature_normalizer(train, split="train")
    cold = soft_observation_posterior(
        train, normalizer=normalizer, temperature=0.1
    )
    warm = soft_observation_posterior(
        train, normalizer=normalizer, temperature=0.5
    )
    for posterior in (cold, warm):
        assert posterior.dtype == np.float64
        assert np.isfinite(posterior).all()
        assert np.all(posterior >= 0.0)
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(np.diag(posterior) >= posterior.max(axis=1) - 1e-15)
        validate_posterior_matrix(posterior, size=4)
    assert warm[0, -1] > cold[0, -1]
    np.testing.assert_array_equal(
        soft_observation_posterior(
            train, normalizer=normalizer, temperature=0.2
        ),
        soft_observation_posterior(
            train, normalizer=normalizer, temperature=0.2
        ),
    )


def test_soft_posterior_ties_are_equal_and_config_freezes_temperatures():
    signatures = np.asarray(
        [
            [0, 1, 2, 3, 4, 0, 1],
            [0, 1, 2, 3, 4, 0, 1],
            [2, 3, 4, 5, 6, 1, 0],
        ],
        dtype=np.float32,
    )
    normalizer = fit_signature_normalizer(signatures, split="train")
    posterior = soft_observation_posterior(
        signatures, normalizer=normalizer, temperature=0.2
    )
    assert posterior[0, 0] == posterior[0, 1]
    assert posterior[1, 0] == posterior[1, 1]
    np.testing.assert_array_equal(posterior[0], posterior[1])

    config = load_observation_posterior_config(CONFIG)
    assert config.default_temperature == 0.2
    assert config.supported_temperatures == (0.1, 0.2, 0.5)


def test_observable_digest_uses_observation_content_without_identity():
    shape = (4, 4)
    visible = np.zeros(shape, dtype=bool)
    visible[1, 1] = True
    occupied = visible.copy()
    observation = CounterfactualObservation(
        visible_mask=visible,
        visible_occupied_mask=occupied,
        visible_dynamic_occupancy=occupied.copy(),
        newly_visible_mask=visible.copy(),
        updated_age_map=np.zeros(shape, dtype=np.float32),
    )
    same = CounterfactualObservation(
        visible_mask=visible.copy(),
        visible_occupied_mask=occupied.copy(),
        visible_dynamic_occupancy=np.zeros(shape, dtype=bool),
        newly_visible_mask=np.zeros(shape, dtype=bool),
        updated_age_map=np.ones(shape, dtype=np.float32),
    )
    changed_occupied = np.zeros(shape, dtype=bool)
    changed = CounterfactualObservation(
        visible_mask=visible.copy(),
        visible_occupied_mask=changed_occupied,
        visible_dynamic_occupancy=changed_occupied.copy(),
        newly_visible_mask=visible.copy(),
        updated_age_map=np.zeros(shape, dtype=np.float32),
    )
    assert observable_observation_digest(observation) == observable_observation_digest(
        same
    )
    assert observable_observation_digest(observation) != observable_observation_digest(
        changed
    )


def test_posterior_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        exact_observation_posterior(("ok", ""))
    signatures = np.zeros((2, 7), dtype=np.float32)
    signatures[0, 0] = np.nan
    clean = np.zeros((2, 7), dtype=np.float32)
    normalizer = fit_signature_normalizer(
        np.asarray([[0] * 7, [1] * 7], dtype=np.float32), split="train"
    )
    with pytest.raises(ValueError, match="finite"):
        soft_observation_posterior(
            signatures, normalizer=normalizer, temperature=0.2
        )
    with pytest.raises(ValueError, match="positive"):
        soft_observation_posterior(
            clean, normalizer=normalizer, temperature=0.0
        )
    with pytest.raises(ValueError, match="shape"):
        validate_posterior_matrix(np.eye(2, dtype=np.float64), size=3)
