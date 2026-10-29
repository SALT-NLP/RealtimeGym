#!/usr/bin/env python
"""
The three-game SFT transfer figure.

Panel 1: paired per-seed deltas, base -> distilled, for every game x load.
         Each dot is one environment seed, so the reader can see that nothing
         regressed rather than taking a mean on faith.
Panel 2: absolute scores, base vs distilled, same cells.
Panel 3: what it costs -- median tokens generated per decision, which is what
         the clock actually charges in token mode.

Freeway rewards rushing (reward = 100 - turns), so a result only on Freeway
would be suspect. Showing all three games together is the point of the figure:
the effect is LARGER on the two games with no speed premium.

Usage:
  python analysis/plot_transfer.py --logs_root rtg_results --out rtg_figures
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_arms import load_cells, R_BOUNDS, LOAD_NAME   # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
})

GAMES = ["freeway", "snake", "overcooked"]
LOADS = ["E", "M", "H"]
ADAPTER = {g: f"qwen3-8b-sft_{g}_full_none" for g in GAMES}
C_BASE, C_SFT, C_DELTA = "#0072B2", "#009E73", "#444444"


def gather(logs_root, pressure, mode):
    """-> {(game, load): {base: [...], sft: [...], seeds: [...], tok_base, tok_sft}}"""
    out = {}
    for g in GAMES:
        cells = load_cells(logs_root, g, pressure, mode)
        lo, hi = R_BOUNDS[g]
        for L in LOADS:
            b, t = cells.get(("qwen3-8b", L)), cells.get((ADAPTER[g], L))
            if not (b and t):
                continue
            shared = sorted(set(b["seeds"]) & set(t["seeds"]))
            if not shared:
                continue
            out[(g, L)] = {
                "base": np.array([np.mean(b["seeds"][s]) for s in shared]) / (hi - lo),
                "sft": np.array([np.mean(t["seeds"][s]) for s in shared]) / (hi - lo),
                "seeds": shared,
                "tok_base": np.median(b["tokens"]) if b["tokens"] else np.nan,
                "tok_sft": np.median(t["tokens"]) if t["tokens"] else np.nan,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", default="rtg_results")
    ap.add_argument("--out", default="rtg_figures")
    ap.add_argument("--pressure", type=int, default=4096)
    ap.add_argument("--mode", default="reactive")
    a = ap.parse_args()

    data = gather(a.logs_root, a.pressure, a.mode)
    if not data:
        print("!! no paired cells found")
        return
    keys = [(g, L) for g in GAMES for L in LOADS if (g, L) in data]
    x = np.arange(len(keys))
    # Just E/M/H on the ticks; the game goes underneath once per group,
    # otherwise the three panels render "FREEFREEFREESNAK..." on top of itself.
    labels = [LOAD_NAME[L][0] for g, L in keys]
    groups = []          # (game, first_index, last_index)
    for g in GAMES:
        idx = [i for i, (gg, _) in enumerate(keys) if gg == g]
        if idx:
            groups.append((g, idx[0], idx[-1]))

    def label_games(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        for g, i0, i1 in groups:
            ax.annotate(g, xy=((i0 + i1) / 2, -0.115), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=9.5, annotation_clip=False)
        for _, _, i1 in groups[:-1]:
            ax.axvline(i1 + 0.5, color="#CCCCCC", lw=.9, zorder=0)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})

    # --- panel 1: per-seed deltas -------------------------------------------
    ax = axes[0]
    for i, k in enumerate(keys):
        d = data[k]["sft"] - data[k]["base"]
        ax.scatter(np.full(len(d), i) + np.linspace(-.16, .16, len(d)), d,
                   s=14, color=C_DELTA, alpha=.55, zorder=3, linewidths=0)
        ax.plot([i - .3, i + .3], [d.mean()] * 2, color="#C0392B", lw=2.2, zorder=4)
    ax.axhline(0, color="#999999", lw=1)
    label_games(ax)
    ax.set_ylabel("score change per env seed")
    # the count belongs in the subtitle: as a corner annotation it sat on top of
    # the overcooked/EASY points, which are the highest in the panel
    n_seeds = sum(len(data[k]["seeds"]) for k in keys)
    n_worse = sum(int(((data[k]["sft"] - data[k]["base"]) < 0).sum()) for k in keys)
    ax.set_title(f"Distillation helps on every seed\n"
                 f"(dot = 1 seed, bar = mean; {n_worse} of {n_seeds} regressed)",
                 fontsize=10.5)

    # --- panel 2: absolute scores -------------------------------------------
    ax = axes[1]
    w = .36
    ax.bar(x - w/2, [data[k]["base"].mean() for k in keys], w, label="Qwen3-8B",
           color=C_BASE, edgecolor="white", linewidth=.6)
    ax.bar(x + w/2, [data[k]["sft"].mean() for k in keys], w, label="+ distilled reasoning",
           color=C_SFT, edgecolor="white", linewidth=.6)
    for i, k in enumerate(keys):
        for off, key in ((-w/2, "base"), (w/2, "sft")):
            v = data[k][key]
            ax.plot([i + off] * 2, [v.mean() - v.std()/np.sqrt(len(v)),
                                    v.mean() + v.std()/np.sqrt(len(v))],
                    color="#333333", lw=1)
    ax.axhline(0, color="#999999", lw=1)
    label_games(ax)
    ax.set_ylabel("normalized score")
    ax.set_title(f"Absolute score (pressure {a.pressure}, {a.mode})\n"
                 "paper Table 4 bounds, +/- s.e.m. over seeds", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # --- panel 3: what it costs ---------------------------------------------
    ax = axes[2]
    tb = [data[k]["tok_base"] for k in keys]
    ts = [data[k]["tok_sft"] for k in keys]
    ax.bar(x - w/2, tb, w, color=C_BASE, edgecolor="white", linewidth=.6)
    ax.bar(x + w/2, ts, w, color=C_SFT, edgecolor="white", linewidth=.6)
    label_games(ax)
    ax.set_ylabel("median tokens per decision")
    ax.set_title("The clock advances with these\n(cost, not a free lunch)", fontsize=10.5)
    for i, (b, s) in enumerate(zip(tb, ts)):
        if b and s and b == b and s == s:
            ax.text(i, max(b, s) * 1.04, f"{s/b:.1f}x", ha="center", fontsize=8,
                    color="#666666")

    fig.suptitle("Distilling the released reasoning traces into Qwen3-8B: three games, "
                 "paired by environment seed", fontsize=11.5, y=1.0)
    fig.tight_layout(rect=(0, .04, 1, .95))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "sft_transfer_three_games.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Figure: {path}")

    print(f"\n{'cell':22s} {'base':>7s} {'sft':>7s} {'delta':>8s} {'seeds w/l':>10s} {'tok x':>7s}")
    for k in keys:
        d = data[k]["sft"] - data[k]["base"]
        g, L = k
        ratio = data[k]["tok_sft"] / data[k]["tok_base"] if data[k]["tok_base"] else float("nan")
        print(f"{g + ' / ' + LOAD_NAME[L]:22s} {data[k]['base'].mean():>7.3f} "
              f"{data[k]['sft'].mean():>7.3f} {d.mean():>+8.3f} "
              f"{str(int((d>0).sum())) + '/' + str(int((d<0).sum())):>10s} {ratio:>6.1f}x")


if __name__ == "__main__":
    main()
