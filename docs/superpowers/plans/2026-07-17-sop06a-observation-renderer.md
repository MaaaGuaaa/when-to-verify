# SOP06-A Observation Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SOP06 history-only BEV renderer and oracle-isolation tests without waiting for the final SOP05 generated-event artifacts.

**Architecture:** A single pure renderer consumes only a validated `BaseState`, complete scene history/spec mappings, static/programmatic occupancy, a sensor definition, and the frozen base configuration. It rasterizes every historical scene, applies the existing ray caster, derives the frozen two history and nine state channels, and returns safe metadata. It never accepts `OracleContext`, `OracleWorld`, a future trajectory, or a target-specific argument; an empty variant is rendered by deleting the target from the generic scene mappings and calling the same function again.

**Tech Stack:** Python 3.11, NumPy, pytest, existing `src.contracts`, `src.geometry`, and `src.generation.structural_blindspot`; all Python execution through Slurm with the frozen SOP environment.

---

## Scope and contract boundary

Owned files for this phase:

- Create: `src/generation/observation_renderer.py`
- Create: `tests/test_observation_renderer.py`
- Create: `tests/test_input_oracle_isolation.py`
- Create: `docs/superpowers/plans/2026-07-17-sop06a-observation-renderer.md`

Do not modify `src/contracts.py`, `configs/base.yaml`, `DECISIONS.md`, `STATUS.md`,
`pyproject.toml`, any SOP03/SOP05 file, or the existing paired generator in this phase.

The production signature is frozen as:

```python
def render_observation(
    base_state: BaseState,
    *,
    scene_dynamic_history: Mapping[str, np.ndarray],
    scene_dynamic_specs: Mapping[str, dict[str, object]],
    static_occupancy: np.ndarray,
    sensor_config: StructuralBlindSpot | None,
    config: Mapping[str, Any],
) -> RenderedObservation:
    ...
```

`RenderedObservation` has exactly `bev_history`, `state_channels`, and `metadata`.
The two arrays are owned float32 contiguous copies with shapes `[K,2,H,W]` and
`[9,H,W]`. Metadata has exactly:

```text
renderer_layout_version
base_state_id
sensor_config_digest
static_occupancy_digest
```

Contract changes requested from SOP05 before later integration:

- `TransplantedDynamicObject` must expose real `history_poses[8,3]`.
- Main generation must enforce `time_scale_range=[1.0,1.0]`.
- SOP05 must produce the versioned target-motion shard defined by the authoritative spec.

SOP06-A must not create a compatibility layer for the current future-only object.

## Slurm command template

Every RED and GREEN test command uses:

```bash
srun \
  --partition=gpu \
  --job-name=sop06a-tdd \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=4G \
  --time=00:10:00 \
  --kill-on-bad-exit=1 \
  --chdir=/home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI-worktrees/sop-05-06-joint-pairs \
  env PYTHONHASHSEED=0 OMP_NUM_THREADS=2 \
  /home/home/ccnt_zq/zq_zhouyiqun/hyz_ws/AAAI/.conda-envs/sop4-risk/bin/python \
  -m pytest -q \
  tests/test_observation_renderer.py \
  tests/test_input_oracle_isolation.py
```

No GPU resource is requested because the renderer tests are CPU-only.

---

### Task 1: Freeze the safe public API and empty-scene tensor contract

**Files:**

- Create: `tests/test_observation_renderer.py`
- Create: `tests/test_input_oracle_isolation.py`
- Create: `src/generation/observation_renderer.py`

- [x] **Step 1: Write the first failing tests**

Create a 9×9, `K=3`, `resolution=1.0 m` fixture with the current robot at the
origin. The first tests import the wished-for API and assert:

```python
def test_empty_scene_outputs_frozen_shapes_dtype_and_partition(renderer_fixture):
    result = render_observation(**renderer_fixture.empty_inputs)
    assert result.bev_history.shape == (3, len(HISTORY_CHANNELS), 9, 9)
    assert result.state_channels.shape == (len(STATE_CHANNELS), 9, 9)
    assert result.bev_history.dtype == np.float32
    assert result.state_channels.dtype == np.float32
    assert result.bev_history.flags.c_contiguous
    assert result.state_channels.flags.c_contiguous
    assert np.isfinite(result.bev_history).all()
    assert np.isfinite(result.state_channels).all()

    visible = result.bev_history[-1, HISTORY_CHANNELS.index("past_visible_mask")]
    free = result.state_channels[STATE_CHANNELS.index("current_visible_free")]
    occupied = result.state_channels[
        STATE_CHANNELS.index("current_visible_occupied")
    ]
    unknown = result.state_channels[
        STATE_CHANNELS.index("current_unobservable_mask")
    ]
    np.testing.assert_array_equal(free + occupied, visible)
    np.testing.assert_array_equal(free + occupied + unknown, np.ones_like(free))
    assert not np.any((free != 0) & (occupied != 0))
```

```python
def test_renderer_public_api_has_no_oracle_or_future_parameter():
    names = tuple(inspect.signature(render_observation).parameters)
    assert not any(
        token in name
        for name in names
        for token in ("oracle", "world", "future", "trajectory")
    )
    assert_no_oracle_leakage(RenderedObservation)
    assert tuple(field.name for field in fields(RenderedObservation)) == (
        "bev_history",
        "state_channels",
        "metadata",
    )
```

- [x] **Step 2: Run RED through Slurm**

Run the exact Slurm command above. Expected: collection fails because
`src.generation.observation_renderer` does not exist.

- [x] **Step 3: Implement only the API, validation shell, and empty-scene channels**

Define:

```python
RENDERER_LAYOUT_VERSION = "bev_history2_state9_v1"

@dataclass(frozen=True)
class RenderedObservation:
    bev_history: np.ndarray
    state_channels: np.ndarray
    metadata: dict[str, str]
```

Implement strict base/config/grid/static validation, allocate channel arrays by
the frozen `HISTORY_CHANNELS` and `STATE_CHANNELS` ordering, rasterize the
un-inflated physical robot rectangle at `robot_history[-1]`, and broadcast raw
SI `v`/`omega` as float32. `sensor_config=None` invokes the existing full-360
ray caster; a `StructuralBlindSpot` invokes `build_structural_visibility`.

- [x] **Step 4: Run GREEN through Slurm**

Run the exact Slurm command above. Expected: `2 passed`.

---

### Task 2: Rasterize complete scene history and enforce visibility semantics

**Files:**

- Modify: `tests/test_observation_renderer.py`
- Modify: `src/generation/observation_renderer.py`

- [x] **Step 1: Add failing physical-visibility tests**

Use a wall between the current robot and a circle target, plus a visible yawed
rectangle context actor. Assert all of the following on each frame:

```python
dynamic = result.bev_history[:, HISTORY_CHANNELS.index("past_dynamic_occupancy")]
visible = result.bev_history[:, HISTORY_CHANNELS.index("past_visible_mask")]
assert np.all(dynamic <= visible)
assert target_hidden_cells.any()
assert not dynamic[-1, target_hidden_cells].any()
assert dynamic[0, target_visible_cells].any()
```

Add a wrapping spy around the renderer module's real `raycast_visibility` or
`build_structural_visibility`. Capture the real occupancy passed into it, call
the real implementation, and assert the hidden target footprint is present in
internal total occupancy while absent from model occupancy.

Add exact tests that a circle mask is yaw-invariant and a rectangle mask swaps
its long direction when yaw changes from `0` to `π/2`.

- [x] **Step 2: Run RED through Slurm**

Run the exact Slurm command above. Expected: the new dynamic and footprint
assertions fail because scene histories are not rendered yet.

- [x] **Step 3: Implement complete historical scene rasterization**

For every sorted object ID and every history index:

```python
dynamic_occupancy[k] |= rasterize_footprint(
    footprint_from_spec(scene_dynamic_specs[object_id]),
    scene_dynamic_history[object_id][k],
    grid,
)
total_occupancy[k] = static_mask | dynamic_occupancy[k]
visible[k] = visibility(total_occupancy[k], base_state.robot_history[k])
past_dynamic[k] = dynamic_occupancy[k] & visible[k]
```

Reject non-float32/non-finite/non-`[K,3]` histories, unequal history/spec key
sets, missing or altered BaseState context objects, invalid specs, invalid
static occupancy, and a non-origin current robot pose. Require every occupied
BaseState static cell to remain occupied in the passed static/programmatic map.

- [x] **Step 4: Run GREEN through Slurm**

Run the exact Slurm command above. Expected: all current tests pass.

---

### Task 3: Compute last-seen belief, normalized age, and empty re-rendering

**Files:**

- Modify: `tests/test_observation_renderer.py`
- Modify: `tests/test_input_oracle_isolation.py`
- Modify: `src/generation/observation_renderer.py`

- [x] **Step 1: Add failing hand-computed belief tests**

Arrange for the target cell to be visible only at `k=0`, then hidden at `k=1`
and `k=2`. With `history_dt_s=0.2` and `A_max=5.0`, assert:

```python
assert last_seen[target_cell] == 1.0
assert age[target_cell] == pytest.approx(0.08)
assert age[current_visible_cell] == 0.0
assert age[never_visible_cell] == 1.0
assert last_seen[current_visible_static_cell] == 0.0
```

Render the full scene and a second scene with only the target history/spec key
removed. Assert background/static/robot channels remain equal, while the target
cell's past occupancy and last-seen belief are recomputed to zero. This test
must call the renderer twice; it must not edit a previously returned tensor.

- [x] **Step 2: Run RED through Slurm**

Run the exact Slurm command above. Expected: last-seen/age and empty assertions fail.

- [x] **Step 3: Implement belief updates**

Iterate from oldest to current history frame:

```python
last_seen_dynamic[visible[k]] = dynamic_occupancy[k][visible[k]]
last_visible_index[visible[k]] = k
```

Initialize last-seen occupancy to zero and last-visible index to `-1`. At the
end, use the frozen config values:

```python
age = np.full((H, W), never_seen_value, dtype=np.float32)
seen = last_visible_index >= 0
age[seen] = np.minimum(
    (K - 1 - last_visible_index[seen]) * history_dt_s / a_max_s,
    1.0,
)
age[visible[-1]] = visible_value
```

- [x] **Step 4: Run GREEN through Slurm**

Run the exact Slurm command above. Expected: all tests pass.

---

### Task 4: Prove future/oracle isolation, safe metadata, determinism, and failures

**Files:**

- Modify: `tests/test_input_oracle_isolation.py`
- Modify: `tests/test_observation_renderer.py`
- Modify: `src/generation/observation_renderer.py`

- [x] **Step 1: Add failing isolation and validation tests**

Add tests that:

- passing `dynamic_object_future=` raises Python `TypeError`;
- two caller-side worlds with identical renderer inputs and different future
  arrays produce byte-identical renderer arrays and JSON-identical metadata;
- metadata keys are exactly the four-key whitelist and recursively contain no
  forbidden token, ndarray, target ID, object list, visibility truth, or time;
- outputs share no memory with any input array;
- scene mapping insertion order does not affect arrays or metadata;
- repeated calls are elementwise deterministic;
- NaN/Inf, wrong shape/dtype, mismatched keys, missing/altered context, malformed
  sensor config, and out-of-grid historical sensor poses raise explicit errors.

- [x] **Step 2: Run RED through Slurm**

Run the exact Slurm command above. Expected: at least the new metadata/validation
assertions fail before the final hardening implementation.

- [x] **Step 3: Implement minimal hardening**

Build metadata from stable JSON and BLAKE2 digests only. Copy and make output
arrays C-contiguous float32. Never return internal occupancy, actor IDs, poses,
specs, visibility sequences, or caller mappings. Use sorted object IDs for all
scene traversal and reject instead of skipping malformed actors.

- [x] **Step 4: Run GREEN through Slurm**

Run the exact Slurm command above. Expected: all tests pass without warnings.

---

### Task 5: Focused regression, scientific audit, and local commit

**Files:**

- Verify: `src/generation/observation_renderer.py`
- Verify: `tests/test_observation_renderer.py`
- Verify: `tests/test_input_oracle_isolation.py`
- Verify: `docs/superpowers/plans/2026-07-17-sop06a-observation-renderer.md`

- [x] **Step 1: Run focused regression through Slurm**

Use a 20-minute allocation and run:

```text
tests/test_observation_renderer.py
tests/test_input_oracle_isolation.py
tests/test_pair_variants.py
tests/test_structural_blindspot.py
tests/test_contracts.py
```

Expected: all selected tests pass.

- [x] **Step 2: Run deterministic toy repetitions through Slurm**

Run the exact Slurm TDD command above twice in separate allocations. The
determinism tests themselves call the renderer repeatedly and assert exact array
and metadata equality. Expected: both allocations pass with identical test counts.

- [x] **Step 3: Audit scientific invariants**

Confirm from fresh test output:

- exact shapes and float32 dtypes;
- no NaN/Inf;
- binary partition and subset relations;
- hand-computed age values;
- hidden truth excluded from output;
- future truth cannot affect output;
- input arrays are not mutated;
- empty is independently re-rendered;
- identical inputs are deterministic.

- [x] **Step 4: Check Git boundary and commit**

Stage only the four owned files with explicit paths, run cached diff checks, and
create one local commit. Do not use `git add .`, merge, push, or alter Git config.

Formal SOP06 acceptance remains pending the SOP05 target-motion shard, a
10–100 real-event smoke test, and full six-pack visualization review.
