# SOP05 Regime B Milestones

## M1 - Validate Source And Generate Turns

**Input:** authenticated seen-then-occluded Long40 source and fixed config.

**Work:** verify that H0 is visible, preserve the H1..H7 visibility mask, make a
deterministic per-mother RNG stream, sample the truncated normal angle sequence,
and implement future-only rotation about index 7.

**Output:** legal candidate-shaped histories/currents/futures with unchanged
first eight poses.

**Depends on:** finalized Long40 source contract.

## M2 - Filter Future Environment Contact

**Input:** M1 candidate and non-robot scene environment.

**Work:** reject non-finite/out-of-bounds target sweeps and collisions with
static occupancy, occluders, or represented context sweeps.

**Output:** accepted/rejected candidate with one stable reason.

**Depends on:** M1 and existing geometry sweep utilities.

## M3 - Select One Future

**Input:** M1 angle stream and M2 result.

**Work:** select the first legal candidate within 32 attempts, or emit a counted
`no_legal_future` failure. Do not calculate risk here.

**Output:** one selected future or zero scenarios for that mother.

**Depends on:** M2.

## M4 - Hand Off Finalized Scene

**Input:** M3 accepted results/failures and the common release coordinator.

**Work:** pass accepted scene history to SOP6 and retain selected future only on
the oracle side for SOP7; preserve one-to-one mother identity and the shared A+B
cap.

**Output:** deterministic one-scene-per-mother SOP05 entries and accounted
failures.

**Depends on:** M3 and Regime A's shared release interface.
