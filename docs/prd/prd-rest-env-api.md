# PRD — E12: REST Environment API

**Epic Slug**: `rest-env-api`  
**Priority**: P1  
**Sprint**: 7  
**Size**: L

---

## User Story

> **As a** researcher using a Jupyter notebook or custom ML framework,  
> **I want to** drive a FORGE environment over HTTP without Python bindings,  
> **So that** I can use FORGE from any language (Julia, R, Java) or remote machine without installing native extensions.

---

## Problem Statement

The current `demo_ui` API is tightly coupled to the `forge_demo.py` showcase script. External tools cannot control the environment step-by-step. A proper REST API decouples simulation from presentation.

---

## Acceptance Criteria

| # | Given | When | Then |
|---|---|---|---|
| AC1 | Server is running | `POST /api/env/reset {"seed": 42}` | Returns `{"obs": {...}, "info": {...}, "env_id": "uuid"}` in < 10ms |
| AC2 | Valid env_id exists | `POST /api/env/step {"env_id": "...", "action": 1}` | Returns `{"obs", "reward", "terminated", "truncated", "info"}` |
| AC3 | `GET /api/env/{env_id}/render` | At any time | Returns ASCII string + tile grid JSON |
| AC4 | Env_id does not exist | Any env call | Returns `404 {"detail": "env_id not found"}` |
| AC5 | action is out of range | `POST /api/env/step` | Returns `422 {"detail": "action N out of range [0, M]"}` |
| AC6 | Server starts | No args | Swagger UI available at `/docs` with full OpenAPI spec |
| AC7 | 10 concurrent clients send step requests | Under load | All 10 respond successfully; no data cross-contamination |

---

## Out of Scope

- WebSocket streaming (post-beta)
- Authentication / API keys (post-beta)
- Environment pooling / auto-scaling

---

## Success Metrics

- All 7 ACs pass in integration tests
- `POST /api/env/step` p99 latency < 10ms on localhost
- OpenAPI spec passes `openapi-spec-validator`
- 100% of API routes have integration test coverage

---

## Open Questions

1. Should env sessions be stored in memory (dict) or Redis for multi-process scenarios?
2. What is the session TTL before cleanup? (recommend 30 min idle)
3. Should we use FastAPI's existing `main.py` or a separate API server (`api/main.py`)?

---

## API Contract (OpenAPI Sketch)

```yaml
paths:
  /api/env/reset:
    post:
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                seed: {type: integer}
                config: {type: object}
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ResetResponse'

  /api/env/step:
    post:
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                env_id: {type: string, format: uuid}
                action: {type: integer, minimum: 0}
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/StepResponse'
```

---

## Implementation Notes

- Add `api/` module alongside `demo_ui/backend/`
- Use `ForgeGymnasiumEnv` as backend; store instances in `Dict[UUID, ForgeGymnasiumEnv]`
- Observations must be JSON-serializable: convert numpy arrays to nested lists
- Include `env_id` as a `uuid4` string in all responses
- Write `tests/api/test_rest_api.py` with `httpx` async client
