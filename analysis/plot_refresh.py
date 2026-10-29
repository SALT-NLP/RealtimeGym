#!/usr/bin/env python
"""
Figure for the Gap-3 planning-thread refresh A/B.

Three panels, because the score alone does not show whether the mechanism did
anything:

  1. Score per arm, grouped by time pressure, with the crossing count on each bar.
  2. Plan age at decision time (CDF): how stale the advice the agent acted on was.
     This is what refresh is supposed to change; vanilla runs on `main` do not log
     plan_age, so only gap3-code arms appear here.
  3. Cost vs benefit: planning tokens per env step against score. Refresh buys
     freshness by throwing away unrevealed plan and paying for a new generation.

Usage:
  python analysis/plot_refresh.py --logs_root rtg_results/logs --out rtg_figures
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R_LO, R_HI = 0, 89          # Freeway, paper Table 4
ARM_COLOR = {"vanilla": "#555555", "every2": "#009E73", "every3": "#0072B2",
             "every2+carry": "#E69F00", "every3+carry": "#CC79A7"}

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10.5,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
})


def arm_name(cfg):
    if str(cfg.get("refresh_plan", "False")).lower() != "true":
        return "vanilla"
    k = int(float(cfg.get("refresh_every", 0) or 0))
    name = f"every{k}" if k else "on-change"
    if str(cfg.get("refresh_carryover", "False")).lower() == "true":
        name += "+carry"
    return name


def model_tag(args_log, logs_root, default="qwen3-8b"):
    d = os.path.dirname(os.path.abspath(args_log)); stop = os.path.abspath(logs_root)
    while True:
        f = os.path.join(d, "model.txt")
        if os.path.exists(f):
            for line in open(f):
                if line.startswith("model_tag:"):
                    return line.split(":", 1)[1].strip().lower()
        if d == stop or os.path.dirname(d) == d:
            return default
        d = os.path.dirname(d)


def load(logs_root, game="freeway", load_level="M", model="qwen3-8b", partial=False):
    cells = []
    for args_log in sorted(glob.glob(os.path.join(logs_root, "**", "args.log"), recursive=True)):
        cfg, rew, per_seed = {}, [], {}
        for line in open(args_log):
            if line.startswith("seed:"):
                m = re.search(r"seed:\s*(\d+).*reward:\s*(-?[\d.]+)", line)
                if m:
                    rew.append(float(m.group(2)))
                    per_seed.setdefault(int(m.group(1)), []).append(float(m.group(2)))
            elif ":" in line:
                k, _, v = line.partition(":")
                cfg[k.strip()] = v.strip()
        # a partial run may have no finished seed yet; its per-step staleness is
        # still valid, so only complete cells need rewards
        if (not rew and not partial) or cfg.get("game") != game or cfg.get("mode") != "agile":
            continue
        if cfg.get("cognitive_load") != load_level:
            continue
        if not partial and model_tag(args_log, logs_root) != model:
            continue
        d = os.path.dirname(args_log)
        ages, refreshed, steps, tok2 = [], 0, 0, 0.0
        for f in glob.glob(os.path.join(d, "*.csv")):
            df = pd.read_csv(f)
            steps += len(df)
            if "plan_age" in df:
                ages += [a for a in df["plan_age"].dropna()]
            if "refreshed" in df:
                refreshed += int(df["refreshed"].fillna(0).sum())
            if "model2_token_num" in df:
                tok2 += float(df["model2_token_num"].fillna(0).sum())
        r = np.array(rew) if rew else np.array([0.0])
        cells.append(dict(
            arm=arm_name(cfg), pressure=int(float(cfg["time_pressure"])),
            n=len(r), score=float(((r - R_LO) / (R_HI - R_LO)).mean()),
            sem=float(((r - R_LO) / (R_HI - R_LO)).std() / max(np.sqrt(len(r)), 1)),
            crossed=int((r > 0).sum()), ages=ages, per_seed=per_seed,
            refresh_pct=100 * refreshed / max(steps, 1),
            tok_step=tok2 / max(steps, 1), has_age=bool(ages), partial=partial,
        ))
    # pool cells that are the same arm at the same pressure
    merged = {}
    for c in cells:
        k = (c["arm"], c["pressure"])
        if k in merged:
            m = merged[k]
            tot = m["n"] + c["n"]
            m["score"] = (m["score"] * m["n"] + c["score"] * c["n"]) / tot
            m["crossed"] += c["crossed"]; m["ages"] += c["ages"]
            for k, v in c["per_seed"].items(): m["per_seed"].setdefault(k, []).extend(v)
            m["tok_step"] = (m["tok_step"] * m["n"] + c["tok_step"] * c["n"]) / tot
            m["refresh_pct"] = (m["refresh_pct"] * m["n"] + c["refresh_pct"] * c["n"]) / tot
            m["sem"] = max(m["sem"], c["sem"]) * np.sqrt(m["n"] / tot)
            m["n"] = tot
        else:
            merged[k] = dict(c)
    # Error bars: average repetitions within an env seed, then bootstrap over
    # seeds. The 32 runs in a cell are 8 seeds x 4 samples, not 32 independent
    # draws -- samples on one seed share a map and are correlated. This is the
    # paper's own unit of analysis (App. C.2).
    rng = np.random.default_rng(2026)
    for m in merged.values():
        if m["per_seed"]:
            v = np.array([np.mean(x) for x in m["per_seed"].values()]) / (R_HI - R_LO)
            m["score"] = float(v.mean())
            m["seeds"] = len(v)
            if len(v) > 1:
                b = rng.choice(v, size=(5000, len(v))).mean(axis=1)
                lo, hi = np.quantile(b, [0.025, 0.975])
                m["sem"] = float(max(hi - m["score"], m["score"] - lo))
    return list(merged.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", default="rtg_results/logs")
    ap.add_argument("--out", default="rtg_figures")
    ap.add_argument("--load", default="M")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--partial_root", default="rtg_partial/logs",
                    help="checkpoints of killed runs; used for the staleness panel only")
    a = ap.parse_args()

    cells = load(a.logs_root, load_level=a.load, model=a.model)
    # Killed runs checkpointed their logs. Their SCORES are biased (seeds that
    # finish early are the ones that succeeded), but plan_age is a per-step
    # statistic, so a partial run measures staleness just as well. Use them for
    # the staleness panel only, and only where that arm has no complete run.
    if a.partial_root and os.path.isdir(a.partial_root):
        have = {(c["arm"], c["pressure"]) for c in cells if c["has_age"]}
        for c in load(a.partial_root, load_level=a.load, model=a.model, partial=True):
            if c["has_age"] and (c["arm"], c["pressure"]) not in have:
                cells.append(c)
    if not cells:
        raise SystemExit("No agile Freeway cells found.")
    os.makedirs(a.out, exist_ok=True)

    pressures = sorted({c["pressure"] for c in cells if not c["partial"]}, reverse=True)
    arms = sorted({c["arm"] for c in cells if not c["partial"]}, key=lambda s: (s != "vanilla", s))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    # --- 1. score per arm, grouped by pressure -----------------------------
    ax = axes[0]
    w = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        xs, ys, es, lbl = [], [], [], []
        for j, p in enumerate(pressures):
            c = next((c for c in cells if c["arm"] == arm and c["pressure"] == p and not c["partial"]), None)
            if not c:
                continue
            xs.append(j + i * w - 0.4 + w / 2); ys.append(c["score"]); es.append(c["sem"])
            lbl.append(f"{c['crossed']}/{c['n']}")
        if not xs:
            continue
        b = ax.bar(xs, ys, width=w * 0.92, yerr=es, capsize=3,
                   color=ARM_COLOR.get(arm, "#999999"), label=arm, edgecolor="white")
        for rect, t in zip(b, lbl):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + (es[0] or 0) + 0.02,
                    t, ha="center", fontsize=7.5, color="#444444")
    ax.set_xticks(range(len(pressures)))
    ax.set_xticklabels([f"{p//1024}K" for p in pressures])
    ax.set_xlabel("TIME PRESSURE (tokens/step)")
    ax.set_ylabel("SCORE")
    ax.set_title(f"Freeway, load {a.load}: score by arm\n(label = seeds that crossed)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9)

    # --- 2. how stale was the advice ---------------------------------------
    ax = axes[1]
    # Only the tightest pressure is interesting here: at 8K plans already finish
    # inside ~2 steps, so every arm is 0% stale and the bars say nothing.
    age_p = min(c["pressure"] for c in cells if c["has_age"]) if any(c["has_age"] for c in cells) else None
    withage = [c for c in cells if c["has_age"] and c["pressure"] == age_p]
    if withage:
        labels, fr0, fr1, fr2 = [], [], [], []
        for c in sorted(withage, key=lambda c: (-c["pressure"], c["arm"])):
            v = np.array(c["ages"])
            labels.append(c['arm'] + ('\n(partial)' if c['partial'] else ''))
            fr0.append(100 * (v == 0).mean()); fr1.append(100 * (v == 1).mean())
            fr2.append(100 * (v >= 2).mean())
        x = np.arange(len(labels))
        ax.bar(x, fr0, 0.6, label="fresh (age 0)", color="#009E73")
        ax.bar(x, fr1, 0.6, bottom=fr0, label="1 step stale", color="#9BD6C2")
        ax.bar(x, fr2, 0.6, bottom=np.array(fr0) + np.array(fr1),
               label="2+ steps stale", color="#C0392B")
        for i, c in enumerate(sorted(withage, key=lambda c: (-c["pressure"], c["arm"]))):
            ax.text(i, 102, f"{c['refresh_pct']:.0f}% refreshed", ha="center", fontsize=8.5, color="#444444")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("% of decisions"); ax.set_ylim(0, 112)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.text(0.5, -0.22, "(partial) = checkpoint of a run killed at its time cap; per-step staleness is\nunaffected, but its score would be biased, so it is not plotted on the left",
                transform=ax.transAxes, ha="center", fontsize=7.5, color="#888888")
    else:
        ax.text(0.5, 0.5, "no plan_age logged yet", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
    ax.set_title(f"Staleness of the advice acted on\n(at {age_p//1024}K tokens/step)" if age_p else "Staleness", fontsize=10.5)
    ax.grid(axis="x", visible=False)

    # --- 3. cost vs benefit ------------------------------------------------
    ax = axes[2]
    for c in [c for c in cells if not c["partial"]]:
        ax.errorbar(c["tok_step"], c["score"], yerr=c["sem"], marker="o", markersize=9,
                    color=ARM_COLOR.get(c["arm"], "#999999"), capsize=3)
        ax.annotate(f"{c['arm']}\n{c['pressure']//1024}K", (c["tok_step"], c["score"]),
                    textcoords="offset points", xytext=(10, 6 if c["arm"] == "vanilla" else -14),
                    fontsize=8, color="#444444")
    ax.set_xlabel("planning tokens generated per env step")
    ax.set_ylabel("SCORE")
    ax.set_title("Cost of freshness", fontsize=10.5)
    ax.set_xlim(left=0)

    fig.suptitle("Gap 3: re-grounding a stale planning thread", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(a.out, f"gap3_refresh_{a.load}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Figure: {path}")
    print(f"\n{'arm':14s} {'tp':>6} {'n':>4} {'score':>7} {'±sem':>6} {'crossed':>8} {'refresh%':>9} {'tok/step':>9}")
    for c in sorted(cells, key=lambda c: (c["partial"], -c["pressure"], c["arm"])):
        if c["partial"]:
            v = np.array(c["ages"])
            print(f"{c['arm']:14s} {c['pressure']:>6} {'':>4} {'(partial: staleness only)':>25} "
                  f"{c['refresh_pct']:>5.0f}% refreshed, {100*(v>=2).mean():.0f}% of decisions 2+ steps stale")
        else:
            print(f"{c['arm']:14s} {c['pressure']:>6} {c['n']:>4} {c['score']:>7.3f} {c['sem']:>6.3f} "
                  f"{c['crossed']:>4}/{c['n']:<3} {c['refresh_pct']:>8.0f}% {c['tok_step']:>9.0f}")


if __name__ == "__main__":
    main()
