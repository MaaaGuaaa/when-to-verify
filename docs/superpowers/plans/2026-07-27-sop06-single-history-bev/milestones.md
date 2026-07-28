# SOP06 Single-Scene Milestones

## M1 - Receive Finalized SOP5 Entry

**Input:** one accepted A/B scenario and its authenticated scene context.

**Work:** preserve one-to-one sample/mother identity and split model-safe history
render data from oracle-only future data.

**Output:** one validated `Sop06SingleRendererInput` plus a separate SOP7 oracle
handoff.

**Depends on:** completed SOP5 scenario selection.

## M2 - Render History BEV

**Input:** M1 renderer input and base renderer configuration.

**Work:** call the existing history-only renderer exactly once for the entry.

**Output:** one `RenderedObservation` containing only deployment-available
history/state channels.

**Depends on:** M1 and the existing observation renderer.

## M3 - Hand Off To SOP7

**Input:** M2 observation and M1's separate oracle/trajectory data.

**Work:** retain the ID join while keeping oracle fields out of SOP6 model input.

**Output:** a reproducible SOP7 label-build input for the same sample.

**Depends on:** M2 and the existing risk-data builder.
