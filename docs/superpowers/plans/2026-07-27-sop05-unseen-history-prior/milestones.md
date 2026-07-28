# SOP05 Regime A Milestones

## M1 - Validate And Rotate

**Input:** Long40 target motion and the fixed config.

**Work:** Validate the `8 + 32` layout and H0-hidden source identity, create
stable per-mother randomness, and implement all-40-frame rotation around index
7. H1..H7 visibility does not affect source classification.

**Output:** A transformed target preserving identity, footprint, dtype, and
source arrays.

**Depends on:** authenticated SOP05 mother data.

## M2 - Check Candidate Legality

**Input:** M1 transformed target and mother scene geometry.

**Work:** Use production bounds, footprint sweeps, and history visibility to
reject geometry collision, history robot contact, or any visible history frame.

**Output:** One legal result or one stable rejection reason.

**Depends on:** M1 and existing geometry/visibility authorities.

## M3 - Publish One Scenario Per Mother

**Input:** M2, `p_hidden_human=0.30`, and the stable RNG stream.

**Work:** Emit the empty branch or attempt no more than 32 present angles; retain
only the first legal present target.

**Output:** One finalized empty/present scenario or one counted deficit.

**Depends on:** M2.

## M4 - Hand Off Finalized Scenes

**Input:** M3 realizations and failures.

**Work:** Preserve provenance for audit, enforce the shared A+B cap, and expose
the finalized scene to SOP6 without exposing future/oracle fields at its render
boundary.

**Output:** Deterministic SOP05 scenario entries ready for one SOP6 history BEV
per entry; failures remain accounting records.

**Depends on:** M3 and the shared release coordinator.
