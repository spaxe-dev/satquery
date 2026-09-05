# Developer Workflow

Ownership: A frontend (`apps/web`), B api (`apps/api`+backend thin), C agent/core (`packages/agent,core`), D RS tools (`packages/tools,geospatial,models`), E testing/infra. Interfaces (`packages/core`) stable — cross-boundary change needs schema PR + tests. Branches: `feat/<pkg>-<short>`; commits logical (`chore/feat/refactor/test/docs`); PR must run `pytest -q`, `tsc`, demo smoke. No direct `tools.*` imports from api/ui — via registries. CODEOWNERS equivalent: dir owners in this file.
