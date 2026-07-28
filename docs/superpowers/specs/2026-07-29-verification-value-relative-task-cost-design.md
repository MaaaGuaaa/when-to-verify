# Verification Value Relative Task-Cost Design

## Goal

Make positive and negative verification values arise naturally across the
training distribution. Labels must not be shifted, quantile-balanced, or
selected by sign.

The current real-scene path mixes an absolute Long40 TEB task cost
(``J \approx 5.9--7.8`` in the five-scene audit) with
``reject_cost = 0.2``. Consequently, both the pre-verification and
post-verification decisions saturate at rejection, leaving
``value_target = -action_cost`` for every action.

## Decision Loss

For nominal task cost ``J_0 > 0`` and a policy task cost ``J_pi``, define the
dimensionless non-negative task regret

```text
task_regret(nominal) = 0
task_regret(policy) = max(0, J_pi / J_0 - 1)
```

The decision losses become

```text
execute_loss(world) = risk_weight * risk(nominal, world)
policy_loss(policy, world) =
    task_regret(policy) + risk_weight * risk(policy, world)
```

The existing value definition remains unchanged:

```text
BR_before = min(mean(execute_loss), reject_cost)
post_risk = action_cost + mean(min(best_policy_loss, reject_cost))
value_target = BR_before - post_risk
useful_target = int(value_target > 0)
```

Action cost remains the configured duration, distance, and yaw cost. It is
charged exactly once. ``reject_cost`` is now in the same dimensionless
decision-loss space as risk and relative task regret.

## Interfaces and Audit Evidence

The relative-regret conversion belongs in
``src/generation/verification_gt.py`` beside decision-loss construction. A
small pure helper will validate ``J_0 > 0``, finite policy costs, and finite
non-negative output.

``VerificationValueResult`` will additionally retain the unclipped best-policy
loss for each observed scenario. This label-side audit field permits exact
``reject_cost`` sensitivity analysis without recomputing simulation,
observations, posterior inference, or replanning.

Dataset model inputs and the sign rule remain unchanged. Existing toy fixtures
will use the same relative-cost semantics so real and toy paths cannot diverge.

## Calibration

Use only eligible training scenes. Generate 100 groups with the frozen action
library, ``M=8``, exact posterior, four replanning candidates, and deterministic
seeds. Sweep ``reject_cost`` over ``{0.2, 0.3, 0.5}`` from the stored unclipped
losses. Do not select or discard scenes based on resulting label signs.

Choose the smallest value satisfying all of:

- positive fraction is between 10% and 70%;
- both signs persist on an ``M=16`` subset and across three deterministic seeds;
- positive actions have risk reduction greater than action cost;
- values are not determined solely by action identity;
- all values and component losses are finite and contract-consistent.

Freeze the chosen value in ``configs/verification_gt.yaml`` before held-out
generation. Held-out splits report their observed mix but do not influence
calibration.

## Failure Handling

If no candidate ``reject_cost`` produces both signs, do not reduce action cost
or recenter labels. Record whether unclipped post-verification policies remain
worse than rejection. That outcome means the replanning candidate set is
ineffective and requires a separate safe-stop/evasive-candidate design.

## Verification

Add focused unit tests for relative task regret, nominal-loss cancellation,
positive and negative hand-checkable cases, unclipped-loss audit fields, and
the unchanged identity ``value_target = br_before - post_risk``.

Run only focused tests and bounded calibration through Slurm. The calibration
report must include configuration digests, seeds, scenario-bank sizes, sign
counts, per-action sign counts, quantiles, and rejection-selection rates.
