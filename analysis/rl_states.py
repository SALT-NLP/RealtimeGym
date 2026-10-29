#!/usr/bin/env python
"""
Collect (prompt, env snapshot) pairs for rejection sampling / GRPO.

The env has to be forked to score a candidate action, so a state is only
useful if we hold the actual env object -- a rendered board is not enough.
Rather than reconstruct one from the render, this REPLAYS the logged action
sequence through a fresh seeded env, which reproduces the exact states the
policy visited. replay_viewer.py already validated that this is exact
(152/152 games, zero divergence), and every state here is re-verified against
its logged board anyway.

States come from the SFT model's own runs, so they are on-policy: the states
the model actually reaches, including the mid-lane traffic where it fails.

Usage:
  python analysis/rl_states.py --runs rtg_results/logs/sft_freeway_full_none_*_4096 \\
      --out rl_states.pkl
"""
import argparse
import csv
import glob
import os
import pickle
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import realtimegym  # noqa: E402

LOAD_VERSION = {"E": "0", "M": "1", "H": "2"}


def norm(s):
    return "\n".join(l.rstrip() for l in str(s).strip("\n").splitlines())


def load_of(run_dir):
    m = re.search(r"_([EMH])_\d+$", run_dir.rstrip("/"))
    if m:
        return m.group(1)
    m = re.search(r"freeway_([EMH])_\d+_", os.path.basename(run_dir))
    return m.group(1) if m else "M"


def collect_csv(csv_path, load, verify=True):
    """Replay one game; yield a record per step. Stops at the first divergence
    rather than emitting states that do not match what the agent saw."""
    seed_idx = int(os.path.basename(csv_path).split("_")[1].split(".")[0])
    env, real_seed, _ = realtimegym.make(f"Freeway-v{LOAD_VERSION[load]}", seed=seed_idx)
    obs, _ = env.reset()
    out = []
    for i, row in enumerate(csv.DictReader(open(csv_path))):
        logged = norm(row.get("render", ""))
        if verify and logged and norm(obs["state_string"]) != logged:
            return out, i          # diverged; keep only what matched
        prompt = row.get("model1_prompt") or ""
        action = (row.get("action") or "S").strip()
        tok = (row.get("model1_token_num") or "").strip()
        if prompt:
            out.append({"prompt": prompt, "env": env, "load": load, "seed": real_seed,
                        "turn": env.game_turn, "pos": env.pos,
                        "logged_action": action,
                        "logged_tokens": int(float(tok)) if tok else None})
        import copy
        env = copy.deepcopy(env)   # each record keeps its own frozen env
        obs, done, _, _ = env.step(action)
        if done:
            break
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (globs ok)")
    ap.add_argument("--out", default="rl_states.pkl")
    ap.add_argument("--max-per-game", type=int, default=0, help="subsample long games")
    ap.add_argument("--mid-lane-frac", type=float, default=0.6,
                    help="fraction of the set drawn from pos 2..7, where the "
                         "measured failures happen (12-14 collisions/run, max y~5.5)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dirs = [d for pat in a.runs for d in glob.glob(pat) if os.path.isdir(d)]
    recs, diverged, games = [], 0, 0
    for d in sorted(dirs):
        load = load_of(d)
        for f in sorted(glob.glob(os.path.join(d, "*", "*_*.csv"))):
            got, bad = collect_csv(f, load)
            games += 1
            if bad is not None:
                diverged += 1
            if a.max_per_game and len(got) > a.max_per_game:
                got = random.Random(a.seed).sample(got, a.max_per_game)
            recs += got
    print(f"games replayed: {games}  diverged: {diverged}  states: {len(recs)}")

    if a.limit and len(recs) > a.limit:
        rng = random.Random(a.seed)
        mid = [r for r in recs if 2 <= r["pos"] <= 7]
        rest = [r for r in recs if not (2 <= r["pos"] <= 7)]
        n_mid = min(len(mid), int(a.limit * a.mid_lane_frac))
        picked = rng.sample(mid, n_mid) + rng.sample(rest, min(len(rest), a.limit - n_mid))
        rng.shuffle(picked)
        recs = picked
        print(f"  subsampled to {len(recs)} ({n_mid} from the mid lanes)")

    with open(a.out, "wb") as fh:
        pickle.dump(recs, fh)
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")
    print(f"  pos histogram: {dict(sorted(Counter(r['pos'] for r in recs).items()))}")
    print(f"  load mix:      {dict(Counter(r['load'] for r in recs))}")
    print(f"  logged action: {dict(Counter(r['logged_action'] for r in recs))}")


if __name__ == "__main__":
    main()
