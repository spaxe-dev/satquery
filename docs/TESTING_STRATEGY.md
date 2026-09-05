# Testing Strategy

- Unit: schemas (bounds, confidence ranges), planner (5 tasks + refusal), registry (capability routing), tools (each on noise + demo), geo (tiling, compat), fusion IoU.
- Integration: API->orchestrator->tools (5 demos via TestClient), registry health, report round-trip.
- E2E: Demo1-5 scripts (`scripts/final_verification.py` canonical).
- Failure: invalid ext, corrupt, mismatch dims, unsupported, missing modality, model unavailable (VLM off), tool exception, empty query, huge (25MB+), malformed meta. Run `pytest -q` continuously; frontend `tsc --noEmit + next build`.
