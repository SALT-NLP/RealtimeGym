# Gap 3 — Refreshing the stale planning thread

## Problem
In vanilla AgileThinker the planning thread starts one generation over the
observation at turn `t1` and runs it to completion, revealing it token-by-token
across many environment steps. While the world evolves, the plan stays anchored
to `t1` — it goes **stale** and is never re-grounded until it fully finishes.

- Token mode (`time_unit=token`, the reproducible simulation): the full plan is
  generated up front over the start state ([`base.py` `planning_inference`](src/realtimegym/agents/base.py)) then revealed in `time_pressure`-token slices.
- Time mode (`time_unit=seconds`): one streaming call started on the start state
  runs to completion; new observations are never injected.

## Change
`AgileThinker.think()` now starts a **fresh** planning generation grounded on the
*current* observation either when the previous one finished (`gen_text == ""`, as
before) **or** when `should_refresh_plan()` says the live plan has gone stale.

The decision + helpers live on `BaseAgent` so other planning agents can reuse them:
- `configure_refresh(...)` — set the policy.
- `should_refresh_plan(obs)` — staleness check (only while a generation is in flight).
- `_state_distance(prev, cur)` — game-agnostic `1 - difflib ratio` over `state_string`
  (override for a domain-specific metric).

Two triggers (combine freely):
- **Periodic** `refresh_every=K`: re-ground once the live plan is `K` env steps old.
- **Change-based** `refresh_on_change` + `refresh_change_threshold∈[0,1]`: re-ground
  when the observation diverges enough from the state the plan was grounded on.

Two re-grounding modes:
- **Restart-from-scratch** (default): new plan, fresh prompt only.
- **Restart-with-context** `refresh_carryover=True`: seed the new planning prompt
  with the reasoning revealed so far, so the model can reuse what's still valid.

Defaults keep behavior **byte-identical to vanilla** (`refresh_plan=False`), so this
is a clean A/B.

## New CLI flags (agile mode)
```
--refresh_plan                 # master switch
--refresh_every N              # periodic staleness in env steps (0 = off)
--refresh_on_change            # trigger on observation change
--refresh_change_threshold F   # state-string distance in [0,1] (default 0.3)
--refresh_carryover            # restart-with-context instead of from scratch
```
Note: enabling `--refresh_plan` is a no-op unless `--refresh_every > 0` or
`--refresh_on_change` is set.

## Logged metrics (token mode)
Per step in the run CSV:
- `refreshed` — 1 if this step started a refresh, else 0.
- `plan_age` — env steps between the live plan's start turn and now.
`agent.refresh_count` holds the episode total.
A refresh discards the unrevealed remainder of the old plan and pays for a new
planning generation — i.e. the realistic cost of re-grounding shows up directly
in `model2_token_num`.

## A/B example (Freeway, medium load, tight budget)
Add API keys to `.env` first. Then:
```bash
# Baseline (vanilla AgileThinker)
PYTHONPATH=src .venv/bin/python -m realtimegym.agile_eval \
  --game freeway --cognitive_load M --time_pressure 4096 --internal_budget 2048 \
  --mode agile --planning-model-config configs/example-deepseek-v3.2-planning.yaml \
  --reactive-model-config configs/example-deepseek-v3.2-planning.yaml \
  --seed_num 8 --log_dir logs/vanilla

# Refresh every 2 steps, carrying over partial reasoning
PYTHONPATH=src .venv/bin/python -m realtimegym.agile_eval \
  --game freeway --cognitive_load M --time_pressure 4096 --internal_budget 2048 \
  --mode agile --planning-model-config configs/example-deepseek-v3.2-planning.yaml \
  --reactive-model-config configs/example-deepseek-v3.2-planning.yaml \
  --refresh_plan --refresh_every 2 --refresh_carryover \
  --seed_num 8 --log_dir logs/refresh_every2_carry
```
Compare mean reward across seeds (and `plan_age` / `refreshed` distributions).

## Notes / follow-ups
- Implemented for **AgileThinker**. `PlanningAgent` has the same staleness; it can
  reuse `should_refresh_plan` but its `skip_action` plan-indexing needs care.
- Time mode: a refresh starts a new stream and drops the old one; the orphaned
  background call keeps running until it ends (daemon thread). Fine for token-mode
  experiments; revisit if doing wall-clock runs.
- Toward the RL goal: `plan_age` at decision time and the wasted-token cost of a
  refresh are natural ingredients for a staleness-aware reward.

## Tests
`tests/test_agents.py::TestPlanRefresh` covers the decision logic with no LLM/API.
Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agents.py -q --no-cov`
