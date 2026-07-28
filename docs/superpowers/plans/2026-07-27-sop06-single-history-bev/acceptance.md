# SOP06 Single-Scene Acceptance

Use focused Slurm tests only; lint and typecheck are not gates for this handoff.

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00 \
  --job-name=sop05-boundary env PYTHONPATH=. \
  pytest -q tests/test_sop05_scenario_stage_boundary.py
```

```bash
srun --partition=gpu --ntasks=1 --cpus-per-task=1 --mem=3G --time=00:05:00 \
  --job-name=sop06-single env PYTHONPATH=. \
  pytest -q tests/test_sop06_pipeline.py \
  -k 'seen_prior_uses_generic_single_result_renderer_and_sop7_handoff or seen_prior_combined_coordinator_preserves_one_to_one_identity_and_cap or single_publication_rejects_oracle_fields_in_dynamic_specs or single_renderer_rejects_oracle_fields_in_tampered_dynamic_specs'
```

Acceptance conditions:

- An A/B generator result type carries no risk label or collision evidence.
- The core renderer receives only history-safe arguments and is called once per
  accepted entry.
- The release coordinator rejects duplicate mother/sample identities and a
  requested prefix that cannot be filled.
- The renderer does not require a paired sibling or pair-completion marker.
- Oracle future/risk values remain available only after the SOP6 render boundary.
- Regime-A target IDs are absent from renderer histories, and changing a
  Regime-B target pose in any unobserved frame leaves every model-visible BEV
  cell unchanged.

Do not run a production render in this handoff. The later release job must verify
the combined A+B count is no greater than 125000.
