# STATUS

_Project status board for the event-centered blind-spot risk pipeline._

---

## Workspace layout (2026-07-15)

- **New development space = repo root**: `src/` (code), `configs/base.yaml`, `tests/` (`test_contracts.py`, `test_toy_fixture.py`, `fixtures/`), `scripts/00_validate_contracts.py`, plus `docs/`, `STATUS.md`, `DECISIONS.md`, `AGENTS.md`.
- **All prior work is under `legacy/`**: legacy packages (`bev/`, `data_adapters/`, `evaluation/`, `label_generation/`, `occupancy_prediction_sogmp/`, `planners/`, `risk_dataset/`, `risk_model/`, `verification_dataset/`) and their old `scripts/`, `tests/`, `configs/`, mirrored so legacy code still runs from inside `legacy/`.
- **Shared data dirs stay at root** (not moved): `data/`, `outputs/`, `sources/`, `environments/`.
- Move only; nothing deleted. New SOP work happens at the root, not in `legacy/`.

---

## Current state (2026-07-16)

| SOP | Task | Status | Gate |
| --- | --- | --- | --- |
| SOP-00 | Contracts, config, seeding, toy fixture | Done | G0 passed |
| SOP-01 | Group split, leakage audit, manifests | Done | G1 pending until SOP-07 |
| SOP-02 | Geometry / raster / collision / raycasting | Done | G1a passed |
| SOP-03 | THÖR adapter, base states, snippets | Done | G1 pending until SOP-07 |
| SOP-04 | Differential rollout, query maps | Done | G1 pending until SOP-07 |
| SOP-05 | Typed event generation | Done | G1 pending until SOP-07 |
| SOP-06..16 | Generation output → risk → verification → closed loop | Not started | G1–G5 |

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
.conda-envs/sop4-risk/bin/python -m pytest tests/test_occluder_visibility.py tests/test_structural_blindspot.py tests/test_dynamic_object_transplant.py -q
.conda-envs/sop4-risk/bin/python scripts/00_validate_contracts.py --config configs/base.yaml
```

---

## SOP-05 fixture evidence

- Fixed-seed 10-event smoke: 10 accepted of 10 requested; event mix is environment/structural/mixed = 6/3/1.
- Internal finite resampling: 84 attempts, 74 rejected attempts; rejection reasons are 73 `occluder_no_valid_placement` and 1 `structural_visibility_invalid`.
- Shape, dtype, finite-value, context-preservation, visibility-emergence, and deterministic-output checks pass on the toy fixture.
- Real THÖR record smoke is not run in SOP-05 because production generation inputs are assembled by SOP-06; G1 therefore remains pending until SOP-07.

---

## Unblocked next tasks

- SOP-06 — assemble and persist generated worlds without changing the SOP-05 event semantics.
- SOP-07 — run real-record smoke and complete the G1 gate evidence.

Keep SOP-05→06→07 serial; do not mark G1 passed before SOP-07 evidence is complete.

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
