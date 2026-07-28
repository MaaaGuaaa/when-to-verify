# SOP05 Regime A Acceptance

Run only focused Slurm tests; do not gate this work on local execution, ruff, or
mypy.

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00 \
  --job-name=sop05-a env PYTHONPATH=. \
  pytest -q tests/test_sop05_unseen_prior.py
```

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00 \
  --job-name=sop05-boundary env PYTHONPATH=. \
  pytest -q tests/test_sop05_scenario_stage_boundary.py
```

Acceptance conditions:

- Config rejects a prior other than `0.30`, an attempt bound other than `32`, or
  more than one variant per mother.
- The shared release assigns A exactly when H0 is hidden; H1..H7 visibility
  does not change that identity.
- Rotation is all-40-frame, uniform-angle sampling is reproducible, and index 7
  stays fixed.
- Illegal candidates are rejected for the four stated reasons only; future robot
  collision does not reject a candidate.
- A mother produces at most one scenario. Failed present branches remain deficits.
- The result carries no collision evidence or risk class; SOP6 cannot receive
  oracle sampling fields at its renderer input.

No production-scale generation is required for this code handoff. A later release
run must verify that the combined A+B prefix is at most 125000 before publication.
