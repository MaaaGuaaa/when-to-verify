"""Regression for the SOP05 scenario-synthesis / SOP06 rendering boundary."""

from dataclasses import fields


def test_sop5_generators_feed_sop6_renderer_without_risk_labels() -> None:
    import src.generation.sop05_seen_prior as seen_prior
    import src.generation.sop05_unseen_prior as unseen_prior
    import src.generation.sop06_pipeline as pipeline

    assert pipeline.SeenPriorResult is seen_prior.SeenPriorResult
    assert pipeline.UnseenPriorRealization is unseen_prior.UnseenPriorRealization
    result_fields = {field.name for field in fields(seen_prior.SeenPriorResult)}
    assert "continuous_collision_evidence" not in result_fields
    assert "risk_class" not in result_fields
