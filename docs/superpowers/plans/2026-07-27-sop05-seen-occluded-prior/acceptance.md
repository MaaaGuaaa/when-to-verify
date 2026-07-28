# SOP05 Regime B Acceptance

Run focused Slurm checks only. Ruff and mypy are not acceptance gates for this
handoff because the approved Slurm image does not provide them.

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00 \
  --job-name=sop05-b env PYTHONPATH=. \
  pytest -q tests/test_sop05_seen_prior.py
```

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=3G --time=00:05:00 \
  --job-name=sop06-single env PYTHONPATH=. \
  pytest -q tests/test_sop06_pipeline.py \
  -k 'seen_prior_uses_generic_single_result_renderer_and_sop7_handoff or seen_prior_combined_coordinator_preserves_one_to_one_identity_and_cap or single_publication_rejects_oracle_fields_in_dynamic_specs or single_renderer_rejects_oracle_fields_in_tampered_dynamic_specs'
```

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00 \
  --job-name=sop05-boundary env PYTHONPATH=. \
  pytest -q tests/test_sop05_scenario_stage_boundary.py
```

Acceptance conditions:

- The fixed angle prior is zero-mean truncated normal with sigma `pi/12` and
  support `[-pi, pi)`.
- The shared release assigns B exactly when H0 is visible. H7 may be hidden or
  visible, and its actual state remains model-visible.
- History/current are unchanged; only future 32 poses rotate.
- The environment gate rejects only the five named non-robot reasons.
- First legal selection stops within 32 attempts, otherwise produces one
  accounted failure.
- Accepted B results contain no collision evidence or risk class, and one mother
  contributes at most one scene.

No large generation job is required here. A release run must enforce the combined
A+B maximum of 125000 before publication.
