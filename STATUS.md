# STATUS

_Project status board for the event-centered blind-spot risk pipeline._

---

## Workspace layout (2026-07-15)

- **New development space = repo root**: `src/` (code), `configs/base.yaml`, `tests/` (`test_contracts.py`, `test_toy_fixture.py`, `fixtures/`), `scripts/00_validate_contracts.py`, plus `docs/`, `STATUS.md`, `DECISIONS.md`, `AGENTS.md`.
- **All prior work is under `legacy/`**: legacy packages (`bev/`, `data_adapters/`, `evaluation/`, `label_generation/`, `occupancy_prediction_sogmp/`, `planners/`, `risk_dataset/`, `risk_model/`, `verification_dataset/`) and their old `scripts/`, `tests/`, `configs/`, mirrored so legacy code still runs from inside `legacy/`.
- **Shared data dirs stay at root** (not moved): `data/`, `outputs/`, `sources/`, `environments/`.
- Move only; nothing deleted. New SOP work happens at the root, not in `legacy/`.

---

## Current state (2026-07-15)

| SOP | Task | Status | Gate |
| --- | --- | --- | --- |
| SOP-00 | Contracts, config, seeding, toy fixture | Done | G0 passed |
| SOP-01 | Group split, leakage audit, manifests | Not started | G1 |
| SOP-02 | Geometry / raster / collision / raycasting | Not started | G1a |
| SOP-03 | THÖR adapter, base states, snippets | Not started | G1 |
| SOP-04 | Differential rollout, query maps | Not started | G1 |
| SOP-05..16 | Generation → risk → verification → closed loop | Not started | G1–G5 |

> Note: legacy top-level packages (`bev/`, `risk_model/`, ...) exist from prior work and are untouched by SOP-00. See `DECISIONS.md` D1.

---

## G0 contract gate — passed

- 34 tests pass: `tests/test_contracts.py`, `tests/test_toy_fixture.py`.
- Schema round-trips with exact dtype/shape; `allow_pickle=False`.
- Toy risk labels and net verification value `G*` match hand-derived answers.
- Same seed ⇒ bit-identical toy world; different seed ⇒ probe differs.
- Model-input classes carry no oracle fields.
- `scripts/00_validate_contracts.py --config configs/base.yaml` exits 0.

---

## How to run

Use the project interpreter `.conda-envs/sop4-risk/bin/python` (Python 3.10, numpy 1.24, PyYAML).

```bash
.conda-envs/sop4-risk/bin/python -m pytest tests/test_contracts.py tests/test_toy_fixture.py -q
.conda-envs/sop4-risk/bin/python scripts/00_validate_contracts.py --config configs/base.yaml
```

---

## Unblocked next tasks

SOP-00 unblocks these (each owns disjoint files; safe to run up to 4 write-agents in parallel):

- SOP-01 — `src/datasets/split_manager.py`, `scripts/00_make_splits.py`, `configs/data_thor.yaml`
- SOP-02 — `src/geometry/*`
- SOP-04 — `src/planning/{differential_drive,trajectory_sampler,trajectory_filters,query_maps}.py`
- SOP-09 skeleton — `src/models/{bev_encoder,risk_model,losses}.py` using the toy batch

Do not start SOP-03 until the SOP-01 split contract lands; do not start SOP-05→06→07 in parallel (serial generator chain).

---

## Daily sync template

```markdown
## YYYY-MM-DD / SOP-XX
- Completed:
- Produced artifacts:
- Metrics:
- Blockers:
- Interface changes requested:
- Next 24h:
```
