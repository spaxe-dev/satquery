# UX Audit (judge / expert / novice)

- Judge (30s): hero Demo4 must be 1 click. NOW: 9 tabs bury it. FIX: default to Analysis workspace with Demo4 preselected, big [Drop imagery][Ask][Analyze], plan->evidence->answer visible without scroll.
- Expert: needs scale, CRS, GSD, modality, IoU, warnings. NOW: pixel dims only, no scale factor, IoU without context. FIX: show `has_georef/CRS/GSD`, preview scale factor, IoU with threshold legend, heuristic badges.
- Novice: needs obvious input + recoverable errors. NOW: two ports, /api/docs 404, refusal JSON jargon. FIX: single-port note, friendly refusal + "what to try", file-type hints, 4-file cap visible.

`page.tsx` monolith (1093 lines, any-typed, react-query/zod unused, slider mis-wired, grounding letterbox bug) must split: `Workspace, QueryComposer, TraceView, EvidenceView, GroundingView, ReportView, HealthBadge`. Highest-value: split + typed client + Demo4 default + heuristic badges + scale fix.
