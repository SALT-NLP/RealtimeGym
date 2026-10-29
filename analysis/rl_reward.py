#!/usr/bin/env python
"""
Per-decision reward for Freeway, used by both rejection sampling and GRPO.

Why per-decision and not per-episode: one episode is 50-100 steps at ~1900
tokens each, so a GRPO step over 8 rollouts x 64 prompts would be ~50M tokens.
Forking the env instead gives an exact reward for a single action in one
generation, which is ~200x cheaper and targets the failure mode directly.

The failure mode is measured, not assumed: across the SFT runs, failed games
are hit 12-14 times each and never climb past y~5.5 of 9. They are not
truncated -- 0 of 3118 steps lacked a parseable \\boxed{}. So the reward
penalises collisions, rewards progress, and charges for length separately.

    collided                      -> COLLIDED   (-3)
    safe now, cornered within k   -> DOOMED     (-2)
    otherwise                     -> progress   (+1 up / 0 hold / -1 back)
    minus lam * (tokens / budget) in every case

Safety is tiered rather than additive. An earlier additive version gave a
doomed "up" 1 - 1.0 = 0, exactly tying a safe "hold", so the lookahead term
changed no rankings at all. Tiers make any survivable action beat any doomed
one, which beats any collision, and let progress decide only among the safe.

The dead_end term is what makes this learnable. Measured on 200 on-policy
states: 93% of the model's collisions were ALREADY unavoidable when it chose
-- every action collided -- so an immediate-collision reward can only fix 1%
of decisions. With 2-step lookahead, 15.8% have a survivable alternative the
model passed up. That is 16x the signal, for ~3ms of pure-python search.
Depth 3 finds nothing depth 2 misses, so 2 is the default.
"""
import copy
import re

BOXED = re.compile(r"\\boxed\{\s*([UDS])\s*\}")
VALID = "UDS"
COLLIDED = -3.0         # hit a car this step
DOOMED = -2.0           # safe this step, but cornered within LOOKAHEAD steps
LOOKAHEAD = 2
INVALID = -4.0          # worse than any legal action, including a collision


def parse_action(text):
    """Last \\boxed{...} wins -- the traces reason aloud and may mention
    candidate actions before committing to one."""
    m = BOXED.findall(text or "")
    return m[-1] if m else None


def survivable(env, k):
    """Is there any action sequence of length k from here that never collides?
    Exhaustive over 3^k, which is 9 env steps at the default depth."""
    if k <= 0:
        return True
    for a in VALID:
        fork = copy.deepcopy(env)
        _, done, _, collided = fork.step(a)
        if collided:
            continue
        if done or survivable(fork, k - 1):
            return True
    return False


def score(env, completion, tokens=None, budget=4096, lam=0.0, lookahead=LOOKAHEAD):
    """Fork `env`, apply the action in `completion`, and score it.

    `env` is never mutated. Returns (reward, info) so callers can log the parts
    separately -- a reward that moved only because of the length term looks the
    same as one that moved because the action got better.
    """
    action = parse_action(completion)
    if action is None or action not in VALID:
        return INVALID, {"action": None, "valid": False, "collided": False,
                         "dead_end": False, "progress": 0.0, "len_pen": 0.0}

    fork = copy.deepcopy(env)
    before = fork.pos
    _, done, _, collided = fork.step(action)
    # pos counts DOWN to the goal (9 = start, 0 = crossed), and a collision
    # teleports the player back, so progress comes from the action, not a diff
    progress = {"U": 1.0, "S": 0.0, "D": -1.0}[action]
    if not collided and fork.pos == before and action == "U":
        progress = 0.0                      # blocked at the edge, not progress

    # credit assignment: an action that is safe now but corners the agent is
    # the decision that actually loses these games
    dead_end = (not collided) and (not done) and not survivable(fork, lookahead)

    base = COLLIDED if collided else DOOMED if dead_end else progress
    len_pen = lam * (tokens / budget) if (lam and tokens) else 0.0
    return base - len_pen, {"action": action, "valid": True, "collided": bool(collided),
                            "dead_end": bool(dead_end), "progress": progress,
                            "len_pen": len_pen}


def best_of(env, completions, tokens=None, budget=4096, lam=0.0, lookahead=LOOKAHEAD):
    """-> (index, reward, info) of the highest-scoring completion.

    Ties break toward the SHORTER completion, which is what makes rejection
    sampling shorten the policy without the length term having to dominate.
    """
    toks = tokens or [None] * len(completions)
    scored = []
    for i, (c, t) in enumerate(zip(completions, toks)):
        r, info = score(env, c, t, budget, lam, lookahead)
        scored.append((r, -(t or 0), i, info))
    scored.sort(reverse=True)
    r, _, i, info = scored[0]
    return i, r, info


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    import realtimegym

    env, seed, _ = realtimegym.make("Freeway-v1", seed=0)
    env.reset()
    print(f"self-test on seed {seed}, start pos {env.pos}\n")
    for a in "UDS":
        r, info = score(env, f"reasoning...\n\\boxed{{{a}}}", tokens=1900, lam=0.5)
        print(f"  {a}: r={r:+.3f}  {info}")
    r, info = score(env, "I think maybe up?", tokens=1900, lam=0.5)
    print(f"  (no boxed): r={r:+.3f}  {info}")
    r, info = score(env, "first \\boxed{D} ... actually \\boxed{U}", tokens=10)
    print(f"  (two boxed, last wins): r={r:+.3f} action={info['action']}")
    print(f"\n  env untouched: pos={env.pos} turn={env.game_turn}")
