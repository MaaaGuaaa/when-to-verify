# SOP05R M8 Safe-Stop Contract (v2)

## Status and scope

`sop05r_teb_safe_stop_v2` supersedes the unpublished M8 v1
active-revealability definition.  It applies only to M8 audit outputs produced
with `configs/sop05r_m8_safe_stop.yaml`; it does not alter immutable M6/M7
collections or their configuration digests.

## Primary label

An action is `safe_stop_revealable` exactly when all of the following hold:

1. The action is permitted by the M8 config (`stop_scan` is excluded in v2).
2. The target is hidden at the decision seam, evaluated from the action start
   pose and current target pose.
3. The target becomes visible during the executed action trace.
4. The remaining time to the recorded conflict is at least `braking_margin_s`.
5. The action prefix through first visibility, its reactive braking branch, and
   a stationary hold through the original collision time are collision-free.

The action is interrupted at first visibility.  A complete-action trajectory
is retained only as a diagnostic and must not decide the primary label.

## Non-gating diagnostics

Matched-wait visibility, full-action feasibility, and observed same-goal TEB
recovery after stopping are diagnostic fields.  A pedestrian can legitimately
block a route after the robot has safely stopped, so `route_available=false`
must never invalidate `safe_stop_revealable=true`.

## Version gate and migration

Each serialized audit includes `version`, `label_definition_version`, and the
safe-stop configuration digest.  Consumers must call
`validate_teb_safe_stop_audit_payload` and reject any unknown version,
including v1 payloads.  Do not overwrite or mix v1 results with v2 results;
publish v2 under a new artifact root and record both source M6 digest and M8
safe-stop config digest.
