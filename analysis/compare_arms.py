#!/usr/bin/env python
"""
Paired comparison of two arms measured on the same environment seeds.

Built for base-model vs SFT-adapter, but works for any two arms separated by
model tag. The pairing is the point: a cell's 32 runs are 8 seeds x 4 samples,
not 32 independent draws, because samples on one seed share a map. Treating
them as independent understates the variance and invents significance (it
turned a p=0.15 into a p=0.039 on the gap-3 A/B once the pairing was fixed).

So: average the repeats within each env seed, pair the two arms by seed, then
bootstrap and permutation-test the per-seed differences.

Usage:
  python analysis/compare_arms.py --logs_root rtg_results \\
      --base qwen3-8b --treat qwen3-8b-sft_freeway_short_cap \\
      --game freeway --pressure 4096
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_baseline import effective_budget   # noqa: E402  (same cap detection)

R_BOUNDS = {"freeway": (0, 89), "snake": (-1, 15), "overcooked": (0, 56)}
LOAD_NAME = {"E": "EASY", "M": "MEDIUM", "H": "HARD"}


def model_tag(args_log, logs_root, default="qwen3-8b"):
    d = os.path.dirname(os.path.abspath(args_log))
    stop = os.path.abspath(logs_root)
    while True:
        f = os.path.join(d, "model.txt")
        if os.path.exists(f):
            for line in open(f):
                if line.startswith("model_tag:"):
                    return line.split(":", 1)[1].strip().lower()
        if d == stop or os.path.dirname(d) == d:
            return default.lower()
        d = os.path.dirname(d)


def load_cells(logs_root, game, pressure, mode):
    """-> {(model, load): {seeds: {seed: [rewards]}, tokens: [...], capped: bool}}"""
    out, runs = {}, {}
    for al in sorted(glob.glob(os.path.join(logs_root, "**", "args.log"), recursive=True)):
        cfg, per_seed = {}, {}
        for line in open(al):
            line = line.strip()
            if line.startswith("seed:"):
                m = re.search(r"seed:\s*(\d+).*reward:\s*(-?[\d.]+)", line)
                if m:
                    per_seed.setdefault(int(m.group(1)), []).append(float(m.group(2)))
            elif ":" in line and not set(line) == {"-"}:
                k, _, v = line.partition(":")
                cfg[k.strip()] = v.strip()
        if not per_seed or cfg.get("game") != game or cfg.get("mode") != mode:
            continue
        if int(float(cfg.get("time_pressure", 0))) != pressure:
            continue
        key = (model_tag(al, logs_root), cfg.get("cognitive_load", "M"))
        declared = int(float(cfg.get("internal_budget", 0) or 0))
        runs.setdefault(key, []).append(
            (al, per_seed, declared, effective_budget(os.path.dirname(al), declared)))

    # Same rule as the baseline plot: a run silently capped below its declared
    # internal_budget is a different experiment, so it must not be pooled with
    # an honest one. Keep capped runs only when the cell has nothing better --
    # dropping those would delete the arm instead of cleaning it.
    for key, group in runs.items():
        honest = [g for g in group if g[3] == g[2]]
        use = honest or group
        if honest and len(honest) != len(group):
            for al, _, dec, eff in group:
                if eff != dec:
                    print(f"  drop {os.path.basename(os.path.dirname(al))}: "
                          f"capped at {eff}, cell declares {dec}")
        cell = out.setdefault(key, {"seeds": {}, "tokens": [], "capped": not honest})
        for al, per_seed, _, _ in use:
            for s, v in per_seed.items():
                cell["seeds"].setdefault(s, []).extend(v)
            # cost side: what each decision actually spent on the clock
            for f in glob.glob(os.path.join(os.path.dirname(al), "*_*.csv")):
                try:
                    with open(f) as fh:
                        for row in csv.DictReader(fh):
                            v = (row.get("model1_token_num") or "").strip()
                            if v:
                                cell["tokens"].append(int(float(v)))
                except (OSError, ValueError):
                    continue
    return out


def paired(base, treat, lo, hi, rng, iters=20000):
    """-> dict of per-seed paired stats, or None if the arms share no seeds."""
    shared = sorted(set(base) & set(treat))
    if not shared:
        return None
    b = np.array([np.mean(base[s]) for s in shared]) / (hi - lo)
    t = np.array([np.mean(treat[s]) for s in shared]) / (hi - lo)
    d = t - b
    # bootstrap over seeds, not over runs: the seed is the unit of replication
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(iters)])
    # sign-flip permutation: under the null, which arm is "treated" within a
    # seed is arbitrary, so flipping a seed's sign is an equally likely outcome
    flips = rng.choice([-1.0, 1.0], size=(iters, len(d)))
    null = (flips * d).mean(axis=1)
    p = float((np.abs(null) >= abs(d.mean()) - 1e-12).mean())
    return {"n": len(shared), "seeds": shared, "base": b.mean(), "treat": t.mean(),
            "delta": d.mean(), "ci": (np.percentile(boot, 2.5), np.percentile(boot, 97.5)),
            "p": p, "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", default="rtg_results")
    ap.add_argument("--base", required=True, help="model tag of the control arm")
    ap.add_argument("--treat", required=True, help="model tag of the treated arm")
    ap.add_argument("--game", default="freeway")
    ap.add_argument("--mode", default="reactive")
    ap.add_argument("--pressure", type=int, default=4096)
    ap.add_argument("--loads", nargs="*", default=["E", "M", "H"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    lo, hi = R_BOUNDS[a.game]
    cells = load_cells(a.logs_root, a.game, a.pressure, a.mode)
    rng = np.random.default_rng(a.seed)
    tags = sorted({k[0] for k in cells})
    for want in (a.base, a.treat):
        if want not in tags:
            print(f"!! no runs tagged {want!r}. present: {', '.join(tags)}")
            return

    print(f"{a.game} / {a.mode} / pressure {a.pressure}")
    print(f"  control  {a.base}")
    print(f"  treated  {a.treat}\n")
    hdr = (f"{'load':8s} {'n':>3s} {'control':>7s} {'treat':>7s} {'delta':>8s} "
           f"{'95% CI':>18s} {'p':>7s} {'w/l':>7s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for L in a.loads:
        b = cells.get((a.base, L)), cells.get((a.treat, L))
        if not all(b):
            print(f"{LOAD_NAME.get(L, L):8s}  -- one arm not measured")
            continue
        r = paired(b[0]["seeds"], b[1]["seeds"], lo, hi, rng)
        if r is None:
            print(f"{LOAD_NAME.get(L, L):8s}  -- arms share no env seed")
            continue
        rows.append((L, r, b))
        ci = f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
        star = "*" if r["p"] < 0.05 else " "
        print(f"{LOAD_NAME.get(L, L):8s} {r['n']:>3d} {r['base']:>7.3f} {r['treat']:>7.3f} "
              f"{r['delta']:>+8.3f} {ci:>18s} {r['p']:>6.3f}{star} "
              f"{r['wins']}/{r['losses']}")
        if b[0]["capped"] != b[1]["capped"]:
            print(f"{'':8s}     ^ arms differ in effective budget; not a clean A/B")

    print("\ncost per decision (median tokens generated; the clock advances with it)")
    for L, _, (cb, ct) in rows:
        mb = np.median(cb["tokens"]) if cb["tokens"] else float("nan")
        mt = np.median(ct["tokens"]) if ct["tokens"] else float("nan")
        if mb == mb and mt == mt and mb and mt:
            ratio = (f"{mb / mt:.1f}x cheaper" if mt <= mb
                     else f"{mt / mb:.1f}x dearer")
        else:
            ratio = "n/a"
        print(f"  {LOAD_NAME.get(L, L):8s} control {mb:>7.0f}   treat {mt:>7.0f}   {ratio}")

    if rows:
        print("\nw/l counts seeds where the treated arm beat / lost to the control.")
        print("p is a two-sided sign-flip permutation test on the per-seed")
        print("differences; the CI is a bootstrap over seeds, not over runs.")
        print("p = 1.000 with few non-zero deltas is the test being honest:")
        print("one seed moving cannot distinguish a real effect from noise.")


if __name__ == "__main__":
    main()
