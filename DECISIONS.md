# DECISIONS

_Frozen contracts and change process for the event-centered blind-spot risk project._

---

## Change process

- Only the SOP-00 owner may edit `src/contracts.py`, `configs/base.yaml`, `src/utils/`, and this file.
- Any change to a frozen item below requires a new dated entry here, a bumped `SCHEMA_VERSION`, and re-validation via `scripts/00_validate_contracts.py`.
- Other workflows must not silently work around a contract. If blocked, request a change in the task handoff under "Contract changes requested".

---

## 2026-07-15 · SOP-00 initial freeze

### D1. New `src/` package coexists with legacy code

The fresh reimplementation lives under `src/` (`src.contracts`, `src.utils`). The legacy top-level packages (`bev/`, `risk_model/`, `risk_dataset/`, `data_adapters/`, `planners/`, `verification_dataset/`, `evaluation/`) are left untouched. SOP-00 only adds files; it deletes and rewrites nothing. Rationale: the new spec freezes clean contracts, while prior handoffs report unresolved audit/version issues in the legacy pipeline. Reconciliation or reuse of legacy modules is deferred to the SOPs that need them and must go through this file.

### D2. Three-layer state / oracle separation

`BaseState` (observed inputs), `OracleContext` (pedestrian history + future), and `OracleWorld` (counterfactual hidden world) are distinct types. Future/hidden pedestrian data may only live in `OracleContext` / `OracleWorld`. `RiskSample` and `VerificationSample` carry only deployment-available inputs, supervision labels, and provenance. Enforced by `assert_no_oracle_leakage` over `MODEL_INPUT_CLASSES`.

### D3. Channel layout (frozen order)

- `HISTORY_CHANNELS` (per timestep, ×K): `past_dynamic_occupancy`, `past_visible_mask` → 2
- `STATE_CHANNELS` (single frame): 9 channels, order per spec §11.3
- `TRAJECTORY_CHANNELS`: `swept_volume_mask`, `time_to_arrival_map`, `braking_margin_map`, `centerline_map` → 4
- Sample arrays: `bev_history [K,2,H,W]`, `state_channels [9,H,W]`, `trajectory_channels [4,H,W]`.
- Any add/remove/reorder is a schema change.

### D4. Fixed dimensions

- `ROBOT_STATE_DIM = 2` → `(v, omega)`.
- `ACTION_VECTOR_DIM = 3` → `(duration_s, delta_forward_m, delta_yaw_rad)`.
- `QUANTILE_LEVELS = (0.5, 0.8, 0.9, 0.95)`.
- BEV `H = W = 160`, `resolution = 0.1 m`, `K = 8`, `T = 15`, `dt = 0.2 s`. Robot `0.70 × 0.55 m`, inflation `0.15 m`; pedestrian radius `0.30 m`; `age A_max = 5.0 s`.

### D5. Serialization = single `.npz` (arrays + embedded JSON meta), no pickle

`save_dataclass` writes numeric arrays as `arr_N` plus a `meta_json` string array; `load_dataclass` uses `allow_pickle=False`. No Python object arrays are ever stored. `dict[str, ndarray]` fields are namespaced; `tuple` fields round-trip as tuples. Writes use a temp file + atomic rename.

### D6. Determinism and IDs

All randomness derives from `derive_seed(base_seed, *parts)` (BLAKE2b over a canonical string); Python `hash()` is never used. `sample_id` / `pair_group_id` are order-independent digests. Same seed ⇒ bit-identical; different seed ⇒ at least the stochastic probe changes.

### D7. Provenance `code_version = "unversioned"`

The workspace is not a git repo, so provenance records write `code_version = "unversioned"`. Once git is initialized, callers pass the real commit. No fake commit hashes.

### D8. Storage format

First version uses NPZ + JSON (implemented). Zarr/Parquet are optional enhancements and must not block the pipeline (spec §13, plan §3.3 degrade path).

### D9. Toy fixture geometry simplification (test oracle only)

`tests/fixtures/toy_world.py` uses circular robot/pedestrian footprints (radius `0.30 m`, `R_SUM = 0.60 m`) and a `{execute nominal, reject}` candidate set so risk labels and `G*` are hand-derivable. This is an independent test oracle, not a production simplification; production geometry (SOP-02+) uses the full robot rectangle and inflation.

---

## 2026-07-15 · Generic dynamic-object schema v2

### D10. Preserve every non-robot tracked body

The THÖR adapter must retain every valid non-robot rigid body exposed by a
recording. Objects such as `storage`, `cart`, `carrier`, `LO*`, bins, boxes, and
buckets must not be discarded merely because they are not pedestrians or are
temporarily stationary. The adapter assigns one of the stable types `human`,
`carried_object`, or `unknown_dynamic`; it also preserves the raw name/role in
provenance metadata. Object IDs are recording-scoped (`recording_id::body_name`)
so they remain unique across recordings.

Rationale: collision risk depends on occupied space and motion, not semantic
membership in a pedestrian-only allow-list. Temporarily stationary bodies may
move later and must therefore remain available to observed-state and oracle
construction.

### D11. Dynamic-object fields and geometry replace pedestrian-only contracts

Schema `2.0.0` supersedes the pedestrian-only parts of D2 and D4:

- `BaseState` carries sorted `dynamic_object_ids`, visible dynamic-object
  histories, and their frozen type/footprint specs.
- `OracleContext` carries full dynamic-object histories/futures and aligned
  specs; `OracleWorld` carries dynamic-object trajectories and aligned specs.
- Supported specs are JSON-safe `human` circles, `carried_object` rectangles,
  and `unknown_dynamic` circles or rectangles. Raw roles are provenance only and
  are not embedded in the geometry contract.
- Human radius is `0.30 m`, or `0.45 m` for the THÖR `Carrier` human role.
  Non-human rectangles use the 95th-percentile QTM marker extent when at least
  20 valid frames are available; extents are clamped to `[0.05, 3.0] m` and
  fall back to configured type geometry when marker inference is unavailable.
- Snippet libraries are split- and type-scoped:
  `snippets/<split>/<object_type>/`. Kinematic filters are configured per type.

Serialized schema-v1 artifacts are rejected rather than silently upgraded.
They must be regenerated after the v2 producer pipeline is integrated. Split
grouping by human participant remains unchanged and is distinct from the
dynamic-object IDs stored in base states.
