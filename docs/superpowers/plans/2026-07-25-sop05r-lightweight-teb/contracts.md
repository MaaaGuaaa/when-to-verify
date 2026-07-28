# SOP05R long-horizon lightweight TEB contracts

_Non-negotiable interfaces, identities, and causal boundaries for SOP05R v3 long40._

---

## 📚 Document set

- [Long40 system contract](../../../long40_system_contract.md)
- [Full specification](./full-spec.md)
- [Contracts](./contracts.md)
- [Milestones](./milestones.md)
- [Acceptance](./acceptance.md)
- [State](./state.md)

For SOP05R-specific behavior, this file is authoritative. For the shared time layout,
[the Long40 system contract](../../../long40_system_contract.md) wins across SOPs.
Thresholds and release evidence are authoritative in [acceptance.md](./acceptance.md).

## 🏷️ Version boundary

M1 must freeze these exact semantic identities before other implementation begins:

```python
SOP05R_TEB_GENERATOR_VERSION = "obstacle_first_lightweight_teb_v5"
SOP05R_TEB_TEMPLATE_VERSION = "goal_occluder_template_schedule_v3"
SOP05R_TEB_PLANNER_VERSION = "lightweight_teb_planner_v3"
SOP05R_TEB_PLACEMENT_VERSION = "anchored_human_synchronized_long40_v5"
SOP05R_TEB_OCCLUSION_VERSION = "synchronized_window_occlusion_v3"
SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION = (
    "sop05r_nominal_trajectory_collection_v5"
)
SOP05R_TEB_RUN_VERSION = "sop05r_lightweight_teb_generation_run_v4"
SOP05R_TEB_MANIFEST_VERSION = "sop05r_lightweight_teb_manifest_v4"
SOP05R_TEB_SUMMARY_VERSION = "sop05r_lightweight_teb_summary_v5"
SOP05R_TEB_COMPLETION_MARKER_VERSION = (
    "sop05r_lightweight_teb_producer_complete_v4"
)
SOP05R_LONG40_LAYOUT_VERSION = "history8_current7_future32_v1"
SOP05R_LONG40_SCHEMA_VERSION = "4.0.0"
```

The active production chain uses dynamic-object Schema `4.0.0` with eight history
samples, current sample index `7`, and 32 future endpoints. Schema `3.0.0`,
`history8_current7_future15_v1`, 23-sample snippets, and 15-step trajectories are
archival-only. Current production commands must reject them rather than dispatching to a
compatibility path or implicitly converting them.

`lightweight_teb_planner_v3` is a pre-publication long40 correction. No completed v2
collection was published. V3 returns a bounded eight-second full task route and leaves
Schema `4.0.0` decision-relative trajectory extraction to M6.

The CLI mode is explicit:

```text
--generator-mode obstacle_first_teb
```

Historical `legacy` and `obstacle_first` modes may remain as archival records, but they
are outside the current production contract and cannot feed long40 collections.

## 🔒 Causal boundary

### Information matrix

| Information | Static mother planner | Human placement | Label-side revealability | Post-action observed replanner |
| --- | --- | --- | --- | --- |
| source robot pose/control | allowed | allowed | allowed | replaced by action terminal state |
| fixed goal | allowed | allowed | allowed | allowed, unchanged |
| source static occupancy | allowed | allowed | allowed | allowed |
| generated static occluders | allowed | allowed | allowed | allowed |
| human LongMotionSnippet library | forbidden | allowed | allowed | forbidden |
| selected human trajectory | forbidden | allowed | allowed | only if deployment-observed |
| OracleContext future | forbidden | context-collision gate only | label scoring only | forbidden |
| collision anchor/result | forbidden | allowed | allowed | forbidden |
| best verification action | forbidden | forbidden | output only | forbidden |
| realized hidden-world loss | forbidden | forbidden | output only | forbidden |

The static planner call must occur before snippet selection, target anchoring, collision
testing, or revealability testing. Its public request cannot contain fields named
`target`, `human`, `oracle_context`, `conflict`, `collision`, `label`, `risk`, or
`revealability`.

The post-action replanner uses the same optimization engine but a separate causal request
type. It may receive only dynamic information available after the simulated observation;
it must never receive the hidden Oracle trajectory merely because label code knows it.

## 📥 Source-state contract

The encounter starts exactly at:

```python
start_pose = base_state.robot_history[-1]
start_control = base_state.robot_state
```

Implementation must not independently translate or rotate the robot, static map, or
context. A derived decision state may advance naturally along the generated route, but
must preserve:

- `source_base_state_id`
- source split
- source producer evidence
- source-to-decision timeline
- the exact route prefix used to reach the decision state

The source BaseState and OracleContext are immutable. Their canonical digests must be
unchanged after generation.

## 🧱 Static-occluder contract

The shared obstacle union is:

```python
@dataclass(frozen=True)
class RectangleOccluder:
    occluder_id: str
    semantic_type: str
    pose: np.ndarray          # float64 [3], world frame
    length_m: float
    width_m: float


@dataclass(frozen=True)
class CircleOccluder:
    occluder_id: str
    semantic_type: str
    center_xy: np.ndarray     # float64 [2], world frame
    radius_m: float


StaticOccluder = RectangleOccluder | CircleOccluder
```

The M1 template schedule freezes three sampling families:

```text
rectangle = 0.4
l_shape   = 0.4
circle    = 0.2
```

An `l_shape` is a template-level composite, not a third analytic primitive. It is
expanded deterministically into two overlapping `RectangleOccluder` components before
rasterization, clearance checks, and the M3 planner request. Thus M2's primitive
`StaticOccluder` union remains rectangle-or-circle only, while M3 validates every
component of an L-shaped template.

The immutable template config additionally freezes
`relative_yaw_abs_range_deg = [15.0, 45.0]`. M4 samples its sign uniformly and derives
the magnitude from the per-template deterministic seed. This applies to rectangles and
L shapes relative to the start-goal direction; circles remain unoriented. Placement
must recompute its lateral offset after rotation and still satisfy the direct-corridor
intrusion contract below.

Every occluder implementation must provide equivalent free functions:

```python
occluder_bounds(occluder: StaticOccluder) -> tuple[float, float, float, float]
inflate_occluder(occluder: StaticOccluder, margin_m: float) -> StaticOccluder
point_signed_distance(occluder: StaticOccluder, points_xy: np.ndarray) -> np.ndarray
segment_intersects_occluder(
    occluder: StaticOccluder,
    starts_xy: np.ndarray,
    ends_xy: np.ndarray,
    *,
    epsilon_m: float,
) -> np.ndarray
rasterize_occluder(occluder: StaticOccluder, grid: GridSpec) -> np.ndarray
```

Shapes are immutable, finite, positive-sized, and inside the grid. Tangency counts as
intersection within the frozen `epsilon_m`. Rasterization is used for occupancy and
collision validation; analytic segment intersection is the v3 centerline-occlusion
authority.

### Direct-corridor obstruction

Goal/occluder templates must not place an occluder center on the start-goal centerline
by default. Instead, the template sampler chooses a nonzero signed lateral offset on
either side of that line. It accepts a candidate only when the represented obstacle
lightly intrudes into the straight-driving safety corridor.

Let `d_direct` be the minimum analytic signed distance from the direct start-goal
centerline to the occluder, after subtracting the robot's circumscribed footprint radius
and the frozen base robot inflation. Let `c_min` be the lower configured represented
occluder clearance. The direct-corridor intrusion is:

```text
intrusion_m = c_min - d_direct
```

For v3 templates it must satisfy the frozen minimum:

```text
minimum_direct_corridor_intrusion_m = 0.15
```

The actual intrusion may be larger. Thus a straight route violates the required safety
clearance and is rejected. At least one static primitive must analytically intersect the
direct start-goal centerline, while the sampler chooses the largest such nonzero lateral
offset rather than making a centered full-width blockage. M4 must still retain a candidate
only when M3 returns a dynamically feasible, collision-free route with the same goal.

## 🔧 Planner contracts

### Static mother request

```python
@dataclass(frozen=True)
class StaticTebRequest:
    start_pose: np.ndarray               # float32 [3]
    initial_control: np.ndarray          # float32 [2]
    local_goal_world_pose: np.ndarray    # float32 [3]
    static_occupancy: np.ndarray         # binary float32 [H, W]
    occluders: tuple[StaticOccluder, ...]
    base_config: Mapping[str, object]
    planner_config: LightweightTebConfig
```

### Observed post-action request

```python
@dataclass(frozen=True)
class ObservedTebRequest:
    start_pose: np.ndarray
    initial_control: np.ndarray
    local_goal_world_pose: np.ndarray
    static_occupancy: np.ndarray
    occluders: tuple[StaticOccluder, ...]
    observed_dynamic_obstacles: tuple[ObservedDynamicObstacle, ...]
    base_config: Mapping[str, object]
    planner_config: LightweightTebConfig
```

Both APIs call one private optimization engine. `StaticTebRequest` cannot carry dynamic
objects. `ObservedDynamicObstacle` contains only observation-derived current state and
the frozen deployment prediction policy; it does not carry Oracle future arrays.

### Frozen route horizon

The planner configuration freezes:

```text
band_node_count = 21
initial_band_dt_s = 0.25
band_dt_range_s = [0.1, 0.4]
maximum_route_time_s = 8.0
route_sample_dt_s = 0.2
represented_occluder_clearance_range_m = [0.15, 0.75]
bypass_tracking_allowance_m = 0.08
goal_distance_range_m = [4.0, 4.5]
collision_route_path_fraction_range = [0.20, 0.95]
```

There are exactly 20 variable interval durations. Every duration is finite and inside
`band_dt_range_s`; `initial_band_dt_s` is inside the same range. The maximum interval
support must be at least `maximum_route_time_s`, the initialized total band time must not
exceed it, and the ratio `maximum_route_time_s / route_sample_dt_s` must be an exact
positive integer:

```text
20 * 0.4 == 8.0
20 * 0.25 <= 8.0
8.0 / 0.2 == 40
```

`0.15 m` is the only required accepted-route clearance. The `0.08 m` tracking
allowance is applied only while seeding the bypass band so the bounded controller does
not cut inside that required clearance; it is not a second accepted-route safety margin.

### Result

```python
@dataclass(frozen=True)
class PlannedTebRoute:
    planner_version: str
    goal_world_pose: np.ndarray
    band_poses_world: np.ndarray        # float32 [21, 3], includes start and goal
    band_interval_dt_s: np.ndarray      # float32 [20]
    sample_times_s: np.ndarray          # float32 [40], exactly 0.2, ..., 8.0
    sampled_poses_world: np.ndarray     # float32 [40, 3], endpoint poses
    sampled_controls: np.ndarray        # float32 [40, 2]
    goal_arrival_time_s: float
    task_cost: float


@dataclass(frozen=True)
class LightweightTebResult:
    planner_version: str
    route: PlannedTebRoute | None
    goal_world_pose: np.ndarray
    diagnostics: TebDiagnostics
    rejection_reason: str | None
```

The exact requested start is implicit at time `0.0 s` and also appears as the first band
pose. The 40 uniform samples are future endpoints and therefore do not duplicate that
start. One valid public route is returned. V3 evaluates deterministic straight,
single-waypoint left-bypass, and single-waypoint right-bypass initial bands in that
order, then selects the valid route with minimum frozen task cost. The selected route
must:

- start at the exact requested start pose
- bind the requested initial control
- terminate within configured position and heading tolerance of the exact goal
- reach the goal no later than `maximum_route_time_s`
- expose `goal_arrival_time_s` so M5/M6 can keep the collision anchor before route goal
  arrival without imposing a separate decision-suffix horizon gate
- satisfy velocity, acceleration, angular-rate, curvature, and nonholonomic limits
- pass the frozen dense static-occupancy and represented-occluder collision checks;
  clearance margins, rather than an analytic continuous-time solver, provide the
  between-sample safety allowance
- satisfy the configured represented-occluder clearance band
- have finite deterministic band and uniform-sample arrays

The planner task cost has one definition shared by initial and post-action calls. No
verification-specific discount or nominal-route zeroing is allowed.

M3 must not construct query maps or a `LocalTrajectory`. Those are decision-relative
Schema `4.0.0` products and belong to M6.

## ⏱️ Long40 dual-timeline trajectory contract

`PlannedTebRoute` and `LocalTrajectory` have different authorities:

| Property | Full planner route | Model/data suffix |
| --- | --- | --- |
| Type | `PlannedTebRoute` | `LocalTrajectory` |
| Frame | source world frame | decision-local frame |
| Time support | `0.0–8.0 s` | `0.2–6.4 s` after decision |
| Future endpoints | 40 at `0.2 s` spacing | 32 at `0.2 s` spacing |
| Must reach goal | yes | no |
| Query maps | forbidden in M3 | required in M6 |
| Primary use | reachability, anchors, provenance | model input, dataset, risk labels |

M6 samples the selected continuous full route at:

```text
t_decision + 0.2, ..., t_decision + 6.4 s
```

The 40-sample human trajectory is decision-relative: sample index `7` is exactly
`t_decision`, indices `0..7` are the eight history samples, and indices `8..39` are the
32 future endpoints. Robot history before the M4 route start comes from the source
`BaseState`; later samples come from the fixed route prefix. M6 rebases robot history,
target motion, source context, static occupancy, and occluder metadata into one
decision-local frame and emits a concrete decision `BaseState`.

It transforms those poses and controls into the decision frame, builds Schema `4.0.0`
query maps, retains the unchanged world-frame goal, and copies the full-route task cost.
Every time-indexed `LocalTrajectory` and query-map array has leading dimension `32`;
hard-coded 15-step allocation or truncation is forbidden in the v3 branch.
The collision anchor must be strictly before goal arrival. M6 samples the frozen route
over the full model suffix and does not reject a decision merely because that suffix
extends beyond the goal-arrival timestamp.

The positive collision label must remain inside the model domain:

```text
1.2 s <= t_collision - t_decision <= 6.4 s
```

V4 removes the old absolute encounter-time `conflict_time_range_s` contract. Unknown-key
validation must reject that field in a v4 config; collision timing is derived from the
fixed index-7 decision time.

The collision authority evaluates all 32 future intervals rather than truncating at
`3.0 s`. Positive mothers additionally enforce the `1.2 s` lower margin. A first
collision in the final `6.2–6.4 s` interval is valid when continuous
swept-footprint interpolation proves contact. A discrete-only equality at one sampled
endpoint is invalid.

The v4 base/model contract is:

```text
history_steps = 8
current_index = 7
future_steps = 32
future_dt_s = 0.2
future_horizon_s = 6.4
dynamic_object_schema_version = 4.0.0
trajectory_layout_version = history8_current7_future32_v1
```

## 🎯 Collision-anchor contract

One anchor binds:

```python
@dataclass(frozen=True)
class CollisionAnchor:
    route_sample_index: int
    route_time_s: float
    world_position_xy: np.ndarray       # float64 [2]
    snippet_anchor_index: int
    snippet_time_s: float
```

Required invariants:

```text
snippet_time_s == route_time_s within configured tolerance
snippet_anchor_index == 7 + round(route_time_s / 0.2), so robot route endpoint
and human Long40 frame use the same clock; supported indices are 8..39
route anchor is after the route start; the last sampled interval is representable
route path fraction is in `[0.20, 0.95]`
route anchor is strictly before goal arrival; after goal arrival the route is an
explicit fixed-pose, zero-control terminal hold through the 6.4 s suffix
```

The planner cannot receive this object. Anchor search starts only after a static route is
fixed.

## 🔄 Anchored-placement contract

The primary placement is rigid:

```python
transformed_xy[i] =
    anchor.world_position_xy
    + R(theta_rad) @ (source_xy[i] - source_xy[snippet_anchor_index])
```

Headings and velocity vectors rotate by the same `theta_rad`. The resulting anchor must
equal `world_position_xy` within `1e-6 m`.

```python
@dataclass(frozen=True)
class AnchoredHumanPlacement:
    source_snippet_id: str
    anchor: CollisionAnchor
    rotation_rad: float
    translation_xy_m: np.ndarray
    spatial_scale: float
    temporal_scale: float
    history_poses: np.ndarray       # float32 [8, 3], index 7 is current
    current_pose: np.ndarray        # float32 [3], equals history_poses[7]
    future_poses: np.ndarray        # float32 [32, 3]
    provenance: Mapping[str, object]
```

`spatial_scale` is exactly `1.0` in v4. Temporal scaling is allowed only from the frozen
finite config list and only without extrapolating source support. A rigid transform must
preserve all pairwise source-position distances and speed/acceleration magnitudes up to
numeric tolerance.

The angle schedule is finite, deterministic, and digest-bound. M5 evaluates the frozen
one-degree grid, then retains rotations with at least four clear and one blocked
synchronized history samples. It searches decision-visible rotations first, then
decision-hidden rotations. M5 returns immediately on the first valid candidate and does
not use an unbounded random placement loop.

## 👁️ Occlusion contract

The v4 authority is synchronized centerline intersection over the full history window:

```python
@dataclass(frozen=True)
class CenterlineOcclusionWitness:
    version: str
    time_s: float
    sample_index: int
    robot_position_xy: np.ndarray
    target_position_xy: np.ndarray
    blocking_occluder_id: str
```

For every encounter, the history contains exactly eight samples:

```text
count(clear synchronized samples in the 8-frame history) >= 4
count(blocked synchronized samples in the 8-frame history) >= 1
```

Neither the first nor the decision history frame is fixed as the occluded sample. The
persisted witness is the last blocked history sample; decision time remains at index 7.
Robot and target samples must be time synchronized; arbitrary cross-time point pairs are
invalid.

For an initially-hidden stratum, the configured initial sample is blocked. The run
manifest records the requested and observed stratum. Footprint raycasting may be reported
as an audit comparison but cannot override the v3 centerline label.

The event decision time used by verification actions is independent of the occlusion
witness and must leave:

```text
1.2 s <= t_collision - t_decision <= 6.4 s
```

Replanning margin remains a reported downstream diagnostic but is not part of M6 mother
acceptance. The lower bound covers verification action plus braking. The upper bound is
the complete 32-step label horizon, not a three-second sub-window.

## 💥 Collision contract

The anchored sample equality is necessary but not sufficient. Acceptance requires the
existing continuous footprint authority to prove collision and return:

- first continuous collision time
- robot and target poses at first collision
- minimum signed clearance
- route path fraction
- collision point

The accepted first collision must occur between `1.2 s` and `6.4 s` after the fixed
decision time, inclusive, and its route anchor must precede goal arrival. Goal arrival
must remain within the `8.0 s` full-route limit; it does not constrain suffix length.

A collision in the final `6.2–6.4 s` target interval is eligible when the continuous
swept-footprint authority proves first contact in that interval; a discrete-only
equality at the final sample is rejected. Target motion must remain within bounds and
must not intersect the source static map, generated occluders, protected context motion,
or the robot before the accepted first-collision time.

## 📦 Event and trajectory-record contract

The v3 trajectory record contains one nominal plan represented by two time-domain views:

```python
@dataclass(frozen=True)
class Sop05rTebTrajectoryRecord:
    event_id: str
    source_base_state_id: str
    decision_state_id: str
    template_id: str
    planner_version: str
    config_digest: str
    shared_goal_world_pose: np.ndarray
    full_route: PlannedTebRoute
    nominal_trajectory: LocalTrajectory
```

`full_route` and `nominal_trajectory` are two time-domain views of one nominal plan, not
alternative routes. There are no `alternative_trajectory_ids` and no mandatory stop
route in this record. Current consumers require the exact v3 generator, Schema `4.0.0`,
long40 layout, and collection identity. V1 and pre-long40 v2 records are archival and
must not enter the current SOP06, training, evaluation, or publication path.

Every `GeneratedEvent`/`OracleWorld` metadata envelope binds:

- all semantic versions
- source and decision identities
- source split and producer evidence
- goal and occluder geometry
- planner diagnostics and nominal trajectory ID
- full-route arrays, arrival time, task cost, and semantic digest
- decision-relative suffix arrays and query-map digest
- snippet, anchor, rotation, and temporal-scale evidence
- visibility start result and selected witness
- collision evidence
- revealability evidence
- config digest, seed, source-code identity, and artifact checksums

No v3 completion marker is written unless strict self-reload reproduces the same semantic
digest and all requested quotas are met.

## 🔍 Verification contract

The active-revealability audit is label-side only. A moving action is active-revealable
only if:

- its trace is physically feasible
- it does not collide with static, context, or target motion
- it sees the target earlier than a matched-duration wait by the frozen lead
- first visibility leaves the frozen braking and replanning margin
- the observed post-action planner reaches the unchanged goal
- the realized hidden-world loss is computed only after planner output is fixed

`stop_scan` is never counted as an active moving action. Selection filtering is allowed
only for training. Validation and test collections report natural revealability without
filtering.

All actions are evaluated on the same original decision-relative `0.0–6.4 s` target
timeline. If an action terminates at `t_decision + d`, only the remaining
`6.4 s - d` target support is available for risk and realized-loss evaluation. The
implementation must not synthesize or request a fresh 6.4-second Oracle target future
from the action terminal state.

## 🚫 Compatibility and failure rules

- the current production path accepts only Schema `4.0.0` long40 artifacts
- historical SOP05/SOP05R v1 and pre-long40 v2 artifacts are archival-only
- current commands must not fallback to v1/v2 generation or loaders
- no truncation, padding, endpoint repetition, or extrapolation may convert `15` steps
  into `32` steps
- unknown config keys, missing keys, NaN/Inf, booleans used as numeric fields, and
  unordered ranges fail closed
- rejection reasons belong to a frozen finite vocabulary
- partial output is diagnostic only and never receives a completion marker
- target-scale generation is forbidden until every applicable gate in
  [acceptance.md](./acceptance.md) passes
