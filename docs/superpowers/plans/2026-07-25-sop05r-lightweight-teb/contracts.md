# SOP05R lightweight TEB contracts

_Non-negotiable interfaces, identities, and causal boundaries for SOP05R v2._

---

## 📚 Document set

- [Full specification](./full-spec.md)
- [Contracts](./contracts.md)
- [Milestones](./milestones.md)
- [Acceptance](./acceptance.md)
- [State](./state.md)

If prose elsewhere conflicts with this file, this file wins for implementation. Thresholds
and release evidence are authoritative in [acceptance.md](./acceptance.md).

## 🏷️ Version boundary

M1 must freeze these exact semantic identities before other implementation begins:

```python
SOP05R_TEB_GENERATOR_VERSION = "obstacle_first_lightweight_teb_v2"
SOP05R_TEB_TEMPLATE_VERSION = "goal_occluder_template_schedule_v2"
SOP05R_TEB_PLANNER_VERSION = "lightweight_teb_planner_v2"
SOP05R_TEB_PLACEMENT_VERSION = "anchored_human_rotation_v1"
SOP05R_TEB_OCCLUSION_VERSION = "synchronized_centerline_occlusion_v1"
SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION = (
    "sop05r_nominal_trajectory_collection_v2"
)
SOP05R_TEB_RUN_VERSION = "sop05r_lightweight_teb_generation_run_v1"
SOP05R_TEB_MANIFEST_VERSION = "sop05r_lightweight_teb_manifest_v1"
SOP05R_TEB_SUMMARY_VERSION = "sop05r_lightweight_teb_summary_v1"
SOP05R_TEB_COMPLETION_MARKER_VERSION = (
    "sop05r_lightweight_teb_producer_complete_v1"
)
```

The dynamic-object schema remains `3.0.0`; this redesign changes generation semantics,
planner semantics, trajectory records, and publication identities. V1 and v2 collections
must be rejected across dispatch boundaries rather than implicitly converted.

`lightweight_teb_planner_v2` is a pre-publication correction to the M1 contract. The
previous planner component identity described a planner that directly returned a
three-second `LocalTrajectory`; no artifact using that identity was published. V2
instead returns a bounded full task route and leaves Schema `3.0.0` trajectory extraction
to M6.

The CLI mode is explicit:

```text
--generator-mode obstacle_first_teb
```

Existing `legacy` and `obstacle_first` modes retain their existing meanings.

## 🔒 Causal boundary

### Information matrix

| Information | Static mother planner | Human placement | Label-side revealability | Post-action observed replanner |
| --- | --- | --- | --- | --- |
| source robot pose/control | allowed | allowed | allowed | replaced by action terminal state |
| fixed goal | allowed | allowed | allowed | allowed, unchanged |
| source static occupancy | allowed | allowed | allowed | allowed |
| generated static occluders | allowed | allowed | allowed | allowed |
| human MotionSnippet library | forbidden | allowed | allowed | forbidden |
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
collision validation; analytic segment intersection is the v2 centerline-occlusion
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

For v2 templates it must satisfy the frozen range:

```text
direct_corridor_intrusion_range_m = [0.05, 0.15]
```

Thus a straight route violates the required safety clearance and is rejected, but the
obstacle is only a shallow obstruction rather than a centered full-width blockage. M4
must still retain a candidate only when M3 returns a dynamically feasible, collision-free
route with the same goal.

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
band_node_count = 20
initial_band_dt_s = 0.25
band_dt_range_s = [0.1, 0.4]
maximum_route_time_s = 5.0
route_sample_dt_s = 0.2
represented_occluder_clearance_range_m = [0.15, 0.75]
bypass_tracking_allowance_m = 0.08
```

There are exactly 19 variable interval durations. Every duration is finite and inside
`band_dt_range_s`; `initial_band_dt_s` is inside the same range. The maximum interval
support must be at least `maximum_route_time_s`, the initialized total band time must not
exceed it, and the ratio `maximum_route_time_s / route_sample_dt_s` must be an exact
positive integer:

```text
19 * 0.4 >= 5.0
19 * 0.25 <= 5.0
5.0 / 0.2 == 25
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
    band_poses_world: np.ndarray        # float32 [20, 3], includes start and goal
    band_interval_dt_s: np.ndarray      # float32 [19]
    sample_times_s: np.ndarray          # float32 [25], exactly 0.2, ..., 5.0
    sampled_poses_world: np.ndarray     # float32 [25, 3], endpoint poses
    sampled_controls: np.ndarray        # float32 [25, 2]
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
pose. The 25 uniform samples are future endpoints and therefore do not duplicate that
start. One valid public route is returned. V2 uses exactly one deterministic straight
initial band; the required nonzero occluder lateral offset determines the obstacle
repulsion direction. The selected route must:

- start at the exact requested start pose
- bind the requested initial control
- terminate within configured position and heading tolerance of the exact goal
- reach the goal no later than `maximum_route_time_s`
- satisfy velocity, acceleration, angular-rate, curvature, and nonholonomic limits
- pass the frozen dense static-occupancy and represented-occluder collision checks;
  clearance margins, rather than an analytic continuous-time solver, provide the
  between-sample safety allowance
- satisfy the configured represented-occluder clearance band
- have finite deterministic band and uniform-sample arrays

The planner task cost has one definition shared by initial and post-action calls. No
verification-specific discount or nominal-route zeroing is allowed.

M3 must not construct query maps or a `LocalTrajectory`. Those are decision-relative
Schema `3.0.0` products and belong to M6.

## ⏱️ Dual-horizon trajectory contract

`PlannedTebRoute` and `LocalTrajectory` have different authorities:

| Property | Full planner route | Model/data suffix |
| --- | --- | --- |
| Type | `PlannedTebRoute` | `LocalTrajectory` |
| Frame | source world frame | decision-local frame |
| Time support | `0.0–5.0 s` | `0.2–3.0 s` after decision |
| Future endpoints | 25 at `0.2 s` spacing | 15 at `0.2 s` spacing |
| Must reach goal | yes | no |
| Query maps | forbidden in M3 | required in M6 |
| Primary use | reachability, anchors, provenance | model input, dataset, risk labels |

M6 samples the selected continuous full route at:

```text
t_decision + 0.2, ..., t_decision + 3.0 s
```

It transforms those poses and controls into the decision frame, builds the existing query
maps, retains the unchanged world-frame goal, and copies the full-route task cost. When a
requested suffix time is after `goal_arrival_time_s`, its terminal treatment is
deterministic. The suffix is not rejected merely because its final sample precedes goal
arrival, and M3 does not require a separately verified stationary hold after arrival.

The positive collision label must remain inside the model domain:

```text
0 < t_collision - t_decision <= 3.0 s
```

The five-second domain may extend beyond collision to prove same-goal task completion,
but it must not move the collision outside the 15-step label window.

The existing base/model contract remains unchanged:

```text
future_steps = 15
future_dt_s = 0.2
future_horizon_s = 3.0
dynamic_object_schema_version = 3.0.0
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
snippet_anchor_index is internal, not first or last
route anchor is internal, not first or last
route path fraction is inside the frozen acceptance interval
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
    history_poses: np.ndarray
    current_pose: np.ndarray
    future_poses: np.ndarray
    provenance: Mapping[str, object]
```

`spatial_scale` is exactly `1.0` in v2. Temporal scaling is allowed only from the frozen
finite config list and only without extrapolating source support. A rigid transform must
preserve all pairwise source-position distances and speed/acceleration magnitudes up to
numeric tolerance.

The angle schedule is finite, deterministic, and digest-bound. Its coarse step,
refinement radius, refinement step, and maximum refined candidates are config fields.
No unbounded random placement loop is permitted.

## 👁️ Occlusion contract

The v2 authority is synchronized centerline intersection:

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

For a primary initially-visible encounter:

```text
segment(robot(t_start), target(t_start)) intersects no occluder
exists synchronized t_occ in (t_start, t_collision):
    segment(robot(t_occ), target(t_occ)) intersects an occluder
```

One witness is sufficient. The contract does not require consecutive hidden frames,
terminal hidden frames, or full-footprint invisibility. Robot and target samples must be
time synchronized; arbitrary cross-time point pairs are invalid.

For an initially-hidden stratum, the configured initial sample is blocked. The run
manifest records the requested and observed stratum. Footprint raycasting may be reported
as an audit comparison but cannot override the v2 centerline label.

The event decision time used by verification actions is a selected occlusion witness. It
must leave:

```text
t_collision - t_decision
    >= action_duration + braking_margin + replanning_margin
```

The exact terms and thresholds are frozen config fields.

## 💥 Collision contract

The anchored sample equality is necessary but not sufficient. Acceptance requires the
existing continuous footprint authority to prove collision and return:

- first continuous collision time
- robot and target poses at first collision
- minimum signed clearance
- route path fraction
- collision point

The accepted first collision must occur after the selected decision witness and no later
than `3.0 s` after it. Goal arrival may occur later, up to the full-route limit.

Endpoint-only contact at the final route or target sample is rejected. Target motion must
remain within bounds and must not intersect the source static map, generated occluders,
protected context motion, or the robot before the accepted first-collision time.

## 📦 Event and trajectory-record contract

The v2 trajectory record contains one nominal plan represented by two time-domain views:

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
route in this record. Consumers branch explicitly on generator and collection version.
V1 trajectory records retain their old schema and loader.

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

No v2 completion marker is written unless strict self-reload reproduces the same semantic
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

## 🚫 Compatibility and failure rules

- legacy SOP05 code paths remain unchanged
- SOP05R v1 code and artifacts remain readable only through their v1 dispatch
- v2 must not fallback to v1 generation
- unknown config keys, missing keys, NaN/Inf, booleans used as numeric fields, and
  unordered ranges fail closed
- rejection reasons belong to a frozen finite vocabulary
- partial output is diagnostic only and never receives a completion marker
- target-scale generation is forbidden until every applicable gate in
  [acceptance.md](./acceptance.md) passes
