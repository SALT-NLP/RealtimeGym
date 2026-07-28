#!/usr/bin/env python
"""
Analyze & present Gap-3 (planning-thread refresh) A/B results.

Ingests the eval driver's output layout:
  <logs_root>/<setting>_<timestamp>/
      args.log            # arg dump (config) + per-seed result lines
      {r}_{seed}.csv      # per-step trajectory log

From args.log we read the run *config* (game, cognitive_load, time_pressure,
refresh_plan/every/on_change/carryover) and the per-seed final `reward`; from the
CSVs we read the per-step diagnostics (`plan_age`, `refreshed`, `model2_token_num`).
Scores are normalized to [0,1] with the paper's Rmin/Rmax (Table 4).

Figures (analysis/refresh_figures/):
  fig1_score_vs_pressure.png   headline: score vs time pressure, one line/policy
  fig2_score_vs_load.png       score vs cognitive load, one line/policy
  fig3_plan_age_cdf.png        staleness: plan-age distribution by policy
  fig4_refresh_every_sweep.png score vs refresh_every (the U-shape), vs vanilla
Plus summary_table.csv and a paired t-test (vanilla vs each refresh policy).

Usage:
  python analysis/analyze_refresh.py --logs_root logs
  python analysis/analyze_refresh.py --demo        # synthetic data, to preview layout
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

try:
    from scipy import stats  # optional; only for the significance table
except Exception:  # pragma: no cover
    stats = None

# --- paper Table 4 normalization -------------------------------------------
R_BOUNDS = {"freeway": (0, 89), "snake": (-1, 15), "overcooked": (0, 56)}

# --- Okabe-Ito colorblind-safe categorical palette (fixed order, not cycled) -
PALETTE = ["#000000", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#D55E00"]
GRID = "#DDDDDD"
INK = "#222222"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

LOAD_ORDER = {"E": 0, "M": 1, "H": 2}
LOAD_NAME = {"E": "Easy", "M": "Medium", "H": "Hard"}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _to_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def policy_label(cfg):
    if not _to_bool(cfg.get("refresh_plan", "False")):
        return "vanilla"
    parts = []
    if int(float(cfg.get("refresh_every", 0) or 0)) > 0:
        parts.append(f"every={int(float(cfg['refresh_every']))}")
    if _to_bool(cfg.get("refresh_on_change", "False")):
        parts.append("on-change")
    if _to_bool(cfg.get("refresh_carryover", "False")):
        parts.append("carry")
    return "refresh(" + ",".join(parts) + ")" if parts else "refresh(?)"


def parse_args_log(path):
    """Return (config dict, list of per-seed reward floats)."""
    cfg, rewards = {}, []
    with open(path) as f:
        lines = f.read().splitlines()
    in_header = True
    for ln in lines:
        if in_header:
            if ln.strip() == "":
                in_header = False
                continue
            if ": " in ln:
                k, v = ln.split(": ", 1)
                cfg[k.strip()] = v.strip()
        if "reward:" in ln:
            m = re.search(r"reward:\s*(-?\d+(?:\.\d+)?)", ln)
            if m:
                rewards.append(float(m.group(1)))
    return cfg, rewards


def normalize(game, reward):
    lo, hi = R_BOUNDS.get(game, (0, 1))
    return float(np.clip((reward - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else reward


def load_runs(logs_root):
    """Build (scores_df, diag_df) from the eval output tree."""
    score_rows, diag_rows = [], []
    for args_log in glob.glob(os.path.join(logs_root, "**", "args.log"), recursive=True):
        run_dir = os.path.dirname(args_log)
        cfg, rewards = parse_args_log(args_log)
        if cfg.get("mode") != "agile":  # Gap 3 is an agile-mode change
            continue
        game = cfg.get("game", "?")
        load = cfg.get("cognitive_load", "?")
        pressure = int(float(cfg.get("time_pressure", 0) or 0))
        pol = policy_label(cfg)
        for i, r in enumerate(rewards):
            score_rows.append(dict(game=game, load=load, pressure=pressure,
                                   policy=pol, seed=i, score=normalize(game, r)))
        # diagnostics from per-step CSVs
        for csv in glob.glob(os.path.join(run_dir, "*.csv")):
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            row = dict(game=game, load=load, pressure=pressure, policy=pol)
            if "plan_age" in df:
                row["plan_ages"] = df["plan_age"].dropna().tolist()
            if "refreshed" in df:
                row["n_refresh"] = int(df["refreshed"].fillna(0).sum())
            if "model2_token_num" in df:
                row["plan_tokens"] = float(df["model2_token_num"].fillna(0).sum())
            diag_rows.append(row)
    return pd.DataFrame(score_rows), pd.DataFrame(diag_rows)


# ---------------------------------------------------------------------------
# Synthetic demo data (clearly watermarked; for previewing layout only)
# ---------------------------------------------------------------------------
def demo_data():
    rng = np.random.default_rng(0)
    pressures = [32768, 16384, 8192, 4096]
    policies = {
        "vanilla": dict(base=0.9, decay=0.85, age_scale=6.0),
        "refresh(every=2)": dict(base=0.9, decay=0.45, age_scale=2.0),
        "refresh(every=2,carry)": dict(base=0.92, decay=0.38, age_scale=2.0),
    }
    score_rows, diag_rows = [], []
    for pol, p in policies.items():
        for j, pr in enumerate(pressures):
            mean = np.clip(p["base"] - p["decay"] * (j / (len(pressures) - 1)), 0.03, 0.98)
            for seed in range(8):
                sc = float(np.clip(mean + rng.normal(0, 0.05), 0, 1))
                score_rows.append(dict(game="freeway", load="M", pressure=pr,
                                       policy=pol, seed=seed, score=sc))
                ages = rng.poisson(p["age_scale"], size=40).tolist()
                diag_rows.append(dict(game="freeway", load="M", pressure=pr, policy=pol,
                                      plan_ages=ages,
                                      n_refresh=0 if pol == "vanilla" else int(rng.integers(8, 20)),
                                      plan_tokens=float(rng.integers(15000, 40000))))
    # a cognitive-load sweep too (fixed pressure 8k)
    for pol, p in policies.items():
        for load in ("E", "M", "H"):
            mean = {"E": 0.9, "M": 0.65, "H": 0.4}[load] * (1.0 if pol == "vanilla" else 1.15)
            for seed in range(8):
                score_rows.append(dict(game="freeway", load=load, pressure=8192,
                                       policy=pol, seed=seed,
                                       score=float(np.clip(mean + rng.normal(0, 0.05), 0, 1))))
    # a refresh_every sweep at 4k
    for every in (1, 2, 3, 5, 8):
        peak = 1 - abs(every - 2) * 0.09
        for seed in range(8):
            score_rows.append(dict(game="freeway", load="M", pressure=4096,
                                   policy=f"refresh(every={every})", seed=seed,
                                   score=float(np.clip(0.55 * peak + rng.normal(0, 0.04), 0, 1))))
    return pd.DataFrame(score_rows), pd.DataFrame(diag_rows)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def agg(df, by):
    g = df.groupby(by)["score"]
    out = g.agg(["mean", "sem", "count"]).reset_index()
    out["sem"] = out["sem"].fillna(0.0)
    return out


def policy_color(policies):
    order = sorted(policies, key=lambda p: (p != "vanilla", p))  # vanilla first
    return {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(order)}, order


def _watermark(fig, demo):
    if demo:
        fig.text(0.5, 0.5, "DEMO — SYNTHETIC DATA", fontsize=34, color="#CCCCCC",
                 ha="center", va="center", rotation=25, zorder=-1, alpha=0.5)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_score_vs_pressure(scores, out, demo):
    sub = scores[scores["pressure"] > 0]
    # only policies actually swept across pressure (≥2 distinct budgets), so a
    # separate refresh_every sweep at one budget doesn't clutter / cycle colors
    per_pol = sub.groupby("policy")["pressure"].nunique()
    keep = per_pol[per_pol >= 2].index
    sub = sub[sub["policy"].isin(keep)]
    if sub.empty or sub["pressure"].nunique() < 2:
        return
    cmap, order = policy_color(sub["policy"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for pol in order:
        d = agg(sub[sub["policy"] == pol], "pressure").sort_values("pressure", ascending=False)
        if d.empty:
            continue
        x = [f"{int(p/1024)}k" for p in d["pressure"]]
        ax.plot(x, d["mean"], "-o", color=cmap[pol], lw=2, ms=6, label=pol)
        ax.fill_between(x, d["mean"] - d["sem"], d["mean"] + d["sem"],
                        color=cmap[pol], alpha=0.15, lw=0)
    ax.set_xlabel("time pressure (tokens/step)  —  tighter →")
    ax.set_ylabel("normalized score")
    ax.set_ylim(0, 1)
    ax.set_title("Refresh recovers reward lost to staleness under time pressure")
    ax.legend(frameon=False, loc="lower left")
    _watermark(fig, demo)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_score_vs_pressure.png"), bbox_inches="tight")
    plt.close(fig)


def fig_score_vs_load(scores, out, demo):
    sub = scores[scores["load"].isin(LOAD_ORDER)]
    if sub.empty or sub["load"].nunique() < 2:
        return
    cmap, order = policy_color(sub["policy"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for pol in order:
        d = agg(sub[sub["policy"] == pol], "load")
        if d.empty:
            continue
        d = d.sort_values("load", key=lambda s: s.map(LOAD_ORDER))
        x = [LOAD_NAME[l] for l in d["load"]]
        ax.plot(x, d["mean"], "-o", color=cmap[pol], lw=2, ms=6, label=pol)
        ax.fill_between(x, d["mean"] - d["sem"], d["mean"] + d["sem"],
                        color=cmap[pol], alpha=0.15, lw=0)
    ax.set_xlabel("cognitive load")
    ax.set_ylabel("normalized score")
    ax.set_ylim(0, 1)
    ax.set_title("Score vs cognitive load by refresh policy")
    ax.legend(frameon=False, loc="upper right")
    _watermark(fig, demo)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_score_vs_load.png"), bbox_inches="tight")
    plt.close(fig)


def fig_plan_age_cdf(diag, out, demo):
    if diag.empty or "plan_ages" not in diag:
        return
    cmap, order = policy_color(diag["policy"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.6))
    plotted = False
    for pol in order:
        ages = [a for lst in diag[diag["policy"] == pol]["plan_ages"].dropna() for a in lst]
        if not ages:
            continue
        xs = np.sort(ages)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, "-", color=cmap[pol], lw=2, label=pol)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("plan age at decision time (env steps since plan started)")
    ax.set_ylabel("cumulative fraction of decisions")
    ax.set_ylim(0, 1)
    ax.set_title("Refresh caps how stale the acting plan is allowed to get")
    ax.legend(frameon=False, loc="lower right")
    _watermark(fig, demo)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_plan_age_cdf.png"), bbox_inches="tight")
    plt.close(fig)


def fig_refresh_every_sweep(scores, out, demo):
    sweep = scores[scores["policy"].str.match(r"refresh\(every=\d+\)$")].copy()
    if sweep.empty:
        return
    sweep["every"] = sweep["policy"].str.extract(r"every=(\d+)").astype(int)
    d = agg(sweep, "every").sort_values("every")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.errorbar(d["every"], d["mean"], yerr=d["sem"], fmt="-o", color=PALETTE[1],
                lw=2, ms=6, capsize=3, label="refresh(every=k)")
    van = scores[scores["policy"] == "vanilla"]
    if not van.empty:
        m = van["score"].mean()
        ax.axhline(m, color=PALETTE[0], ls="--", lw=1.5, label=f"vanilla ({m:.2f})")
    ax.set_xlabel("refresh_every  k  (env steps)")
    ax.set_ylabel("normalized score")
    ax.set_ylim(0, 1)
    ax.set_title("How often to re-ground: too frequent wastes tokens, too rare stays stale")
    ax.legend(frameon=False, loc="lower center")
    _watermark(fig, demo)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_refresh_every_sweep.png"), bbox_inches="tight")
    plt.close(fig)


def summary_and_stats(scores, out):
    piv = scores.groupby(["game", "load", "pressure", "policy"])["score"].agg(
        ["mean", "sem", "count"]).round(4).reset_index()
    piv.to_csv(os.path.join(out, "summary_table.csv"), index=False)
    print("\n=== Summary (mean score) ===")
    print(piv.to_string(index=False))

    # paired t-test: vanilla vs each refresh policy within each cell (by seed)
    if stats is None:
        print("\n(scipy not available — skipping significance test)")
        return
    print("\n=== Paired t-test vs vanilla (per cell, alpha=0.05) ===")
    rows = []
    for (g, l, pr), cell in scores.groupby(["game", "load", "pressure"]):
        # average any duplicate (seed) entries so pairing is 1:1
        van = cell[cell["policy"] == "vanilla"].groupby("seed")["score"].mean()
        for pol, d in cell.groupby("policy"):
            if pol == "vanilla":
                continue
            r = d.groupby("seed")["score"].mean()
            common = van.index.intersection(r.index)
            if len(common) < 2:
                continue
            t, p = stats.ttest_rel(r.loc[common], van.loc[common])
            rows.append(dict(game=g, load=l, pressure=pr, policy=pol,
                             delta=round(r.loc[common].mean() - van.loc[common].mean(), 3),
                             p_value=round(float(p), 4),
                             sig="*" if p < 0.05 else ""))
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("(no vanilla/refresh pairs sharing seeds yet)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", default="logs")
    ap.add_argument("--out", default="analysis/refresh_figures")
    ap.add_argument("--demo", action="store_true",
                    help="use synthetic data to preview figure layout")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.demo:
        scores, diag = demo_data()
        print("[DEMO MODE] figures use SYNTHETIC data — not real results.")
    else:
        scores, diag = load_runs(args.logs_root)
        if scores.empty:
            print(f"No agile runs found under '{args.logs_root}'. "
                  f"Run the A/B first, or try --demo to preview the figures.")
            return

    fig_score_vs_pressure(scores, args.out, args.demo)
    fig_score_vs_load(scores, args.out, args.demo)
    fig_plan_age_cdf(diag, args.out, args.demo)
    fig_refresh_every_sweep(scores, args.out, args.demo)
    summary_and_stats(scores, args.out)
    print(f"\nFigures written to: {args.out}")


if __name__ == "__main__":
    main()
