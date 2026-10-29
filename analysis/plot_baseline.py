#!/usr/bin/env python
"""
Plot RealtimeGym baseline results in the style of the paper's Figure 5.

Reads the eval driver's output layout:
  <logs_root>/<game>_<load>_<pressure>_<mode>_<ibudget>_<timestamp>/
      args.log          # config block, then one "seed: N reward: R ..." line per seed
      {r}_{seed}.csv    # per-step trajectory

Scores are normalized with the paper's Table 4 bounds: S = (R - Rmin)/(Rmax - Rmin).

Usage:
  python analysis/plot_baseline.py --logs_root logs
  python analysis/plot_baseline.py --logs_root logs --out analysis/figures
"""
import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# paper Table 4
R_BOUNDS = {"freeway": (0, 89), "snake": (-1, 15), "overcooked": (0, 56)}
LOAD_ORDER = {"E": 0, "M": 1, "H": 2}
LOAD_NAME = {"E": "EASY", "M": "MEDIUM", "H": "HARD"}
PALETTE = {"agile": "#009E73", "reactive": "#0072B2", "planning": "#E69F00"}
LABEL = {"agile": "AgileThinker", "reactive": "Reactive", "planning": "Planning"}
MARKER = {"agile": "*", "reactive": "s", "planning": "^"}
MSIZE = {"agile": 15, "reactive": 10, "planning": 11}
MAXLINE = "#C0392B"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
})


def parse_args_log(path):
    """-> (config dict, [rewards])"""
    cfg, rewards = {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or set(line) == {"-"}:
                continue
            if line.startswith("seed:"):
                m = re.search(r"reward:\s*(-?[\d.]+)", line)
                if m:
                    rewards.append(float(m.group(1)))
            elif ":" in line and not line.startswith("seed"):
                k, _, v = line.partition(":")
                cfg[k.strip()] = v.strip()
    return cfg, rewards


def effective_budget(run_dir, declared):
    """The reactive thread generates at min(internal_budget, max_tokens), and
    max_tokens lives in the model config -- which args.log does not record. A
    run whose generations stop dead at a power of two below the declared budget
    was silently capped there, so it is not the experiment it claims to be.

    Detected from the data: a capped run's longest generation equals the cap
    exactly, for every seed.
    """
    best = 0
    for f in glob.glob(os.path.join(run_dir, "*_*.csv")):
        try:
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    v = (row.get("model1_token_num") or "").strip()
                    if v:
                        best = max(best, int(float(v)))
        except (OSError, ValueError):
            continue
    if not best or not declared:
        return declared
    # a cap is a round power of two; a model that merely chose to stop early
    # would not land on one exactly, run after run
    return best if best < declared and best & (best - 1) == 0 else declared


def model_tag(args_log, logs_root, default):
    """The job writes logs/<runtag>/model.txt. args.log only names the config
    file, which is identical for every model (the job patches the model inside),
    so without model.txt all runs would look like the same model."""
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


def load_runs(logs_root, default_model="qwen3-8b"):
    rows = []
    for args_log in sorted(glob.glob(os.path.join(logs_root, "**", "args.log"), recursive=True)):
        cfg, rewards = parse_args_log(args_log)
        if not rewards:
            print(f"  skip (no results yet): {os.path.dirname(args_log)}")
            continue
        # gap3 refresh arms are a separate experiment (see plot_refresh.py); pooling
        # them into "agile" would mix the treatment into its own baseline
        if str(cfg.get("refresh_plan", "False")).lower() == "true":
            continue
        game = cfg.get("game", "freeway")
        lo, hi = R_BOUNDS.get(game, (0, 1))
        declared = int(float(cfg.get("internal_budget", 0) or 0))
        eff = effective_budget(os.path.dirname(args_log), declared)
        for r in rewards:
            rows.append({
                "game": game,
                "load": cfg.get("cognitive_load", "M"),
                "pressure": int(float(cfg.get("time_pressure", 0))),
                "mode": cfg.get("mode", "agile"),
                "model": model_tag(args_log, logs_root, default_model),
                "budget": declared,
                "eff_budget": eff,
                "run": os.path.basename(os.path.dirname(args_log)),
                "reward": r,
                "score": (r - lo) / (hi - lo),
            })
    return drop_capped(rows)


def drop_capped(rows):
    """Within a cell, prefer runs that actually ran at the declared budget.

    Some early runs were silently capped by the model config's max_tokens. They
    declare the same internal_budget as the fixed runs, so pooling them averages
    two different experiments into one point.
    """
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], r["game"], r["mode"], r["load"],
                          r["pressure"], r["budget"]), set()).add(r["eff_budget"])
    drop = {k for k, v in cells.items() if len(v) > 1}
    if not drop:
        return rows
    kept = []
    dropped = {}
    for r in rows:
        k = (r["model"], r["game"], r["mode"], r["load"], r["pressure"], r["budget"])
        if k in drop and r["eff_budget"] != r["budget"]:
            dropped.setdefault(r["run"], [0, r["eff_budget"], r["budget"]])[0] += 1
        else:
            kept.append(r)
    for run, (n, eff, dec) in sorted(dropped.items()):
        print(f"  drop {n:>3} run(s) from {run}: capped at {eff}, cell declares {dec}")
    warn_capped(kept)
    return kept


def warn_capped(rows):
    """Cells with no uncapped run are kept -- dropping them would erase a whole
    model's baseline -- but a curve measured at an effective 2048 cannot be
    compared with one measured at 4096 without saying so."""
    still = {}
    for r in rows:
        if r["eff_budget"] != r["budget"]:
            still.setdefault((r["model"], r["game"], r["mode"]), set()).add(
                (r["eff_budget"], r["budget"]))
    for (model, game, mode), pairs in sorted(still.items()):
        for eff, dec in sorted(pairs):
            print(f"  NOTE {model}/{game}/{mode} ran at an effective {eff} "
                  f"(declares {dec}); not comparable to a {dec} curve")


def pick_slice(rows, vary, fixed):
    """Choose the value of `fixed` that has the most distinct `vary` values.

    The paper holds one axis constant while sweeping the other (Fig. 5: load is
    swept at a fixed 8k pressure; pressure is swept at fixed medium load).
    Pooling across both mixes different cells into one point.
    """
    counts = {}
    for r in rows:
        counts.setdefault(r[fixed], set()).add(r[vary])
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (len(kv[1]), -LOAD_ORDER.get(kv[0], 0)
                                               if fixed == "load" else kv[0]))[0]


def agg(rows, key, fixed=None, fixed_val=None):
    """-> {mode: [(x, mean, std, n), ...]} sorted by x"""
    if fixed is not None and fixed_val is not None:
        rows = [r for r in rows if r[fixed] == fixed_val]
    out = {}
    for row in rows:
        out.setdefault(row["mode"], {}).setdefault(row[key], []).append(row["score"])
    result = {}
    for mode, by_x in out.items():
        pts = [(x, float(np.mean(v)), float(np.std(v)), len(v)) for x, v in by_x.items()]
        result[mode] = sorted(pts, key=lambda t: LOAD_ORDER.get(t[0], t[0]) if key == "load" else t[0])
    return result


def _order(key):
    # Paper convention: both axes read LOW -> HIGH difficulty left to right.
    # For time pressure that means MORE tokens/step on the left (32K ... 4K).
    if key == "load":
        return lambda x: LOAD_ORDER.get(x, 0)
    return lambda x: -x


def _runs(idx):
    """Split sorted axis positions into runs of consecutive integers.
    [0, 2] -> [[0], [1]] (indices into idx); [0, 1, 2] -> [[0, 1, 2]]."""
    runs, cur = [], []
    for i, v in enumerate(idx):
        if cur and v != idx[cur[-1]] + 1:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    return runs


def _plot(ax, series, key, xlabel, reactive_flat=False):
    """Plot every mode on a SHARED x axis, paper-style.

    - union of x values across modes, so modes with different coverage align
    - dotted line = best single-system baseline (reactive / planning) at each x
    - "+N%" = AgileThinker minus that best baseline, in score points x100
      (the paper's +35.0% at medium load is 0.76 - 0.41, an absolute gap)
    - reactive_flat: in token mode the reactive agent never reads time_pressure
      (its output is capped by internal_budget alone), so one measured cell is
      valid at every pressure. Drawn dashed and labelled as such.
    """
    all_x = sorted({p[0] for pts in series.values() for p in pts}, key=_order(key))
    pos = {x: i for i, x in enumerate(all_x)}

    series = {m: list(pts) for m, pts in series.items()}
    flat_note = False
    if reactive_flat and key == "pressure" and len(series.get("reactive", [])) == 1 and len(all_x) > 1:
        _, y, e, n = series["reactive"][0]
        series["reactive"] = [(x, y, e, n) for x in all_x]
        flat_note = True

    for mode in ["planning", "reactive", "agile"]:          # agile drawn on top
        if mode not in series:
            continue
        pts = sorted(series[mode], key=lambda t: pos[t[0]])
        xs = [pos[t[0]] for t in pts]
        ys = [t[1] for t in pts]
        es = [t[2] for t in pts]
        dashed = mode == "reactive" and flat_note
        color = PALETTE.get(mode, "#555555")
        z = 3 if mode == "agile" else 2
        # markers + error bars only; lines are drawn per contiguous run below so
        # a missing cell (e.g. planning at MEDIUM) is a gap, not an interpolation
        ax.errorbar(xs, ys, yerr=es, marker=MARKER.get(mode, "o"),
                    markersize=MSIZE.get(mode, 10), ls="none", capsize=4,
                    elinewidth=1, alpha=0.95, color=color,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=z + 0.5)
        for seg in _runs(xs):
            ax.plot([xs[i] for i in seg], [ys[i] for i in seg], lw=2.2,
                    ls="--" if dashed else "-", color=color, zorder=z)
        # join measured points across unmeasured cells: dashed + faded, so the
        # line reads as continuous without implying a value at the gap
        for k in range(len(xs) - 1):
            if xs[k + 1] - xs[k] > 1:
                ax.plot([xs[k], xs[k + 1]], [ys[k], ys[k + 1]], lw=1.6, ls=(0, (4, 3)),
                        color=color, alpha=0.55, zorder=z - 0.5)
                ax._bridged = True
        ax.plot([], [], marker=MARKER.get(mode, "o"), markersize=MSIZE.get(mode, 10),
                lw=2.2, ls="--" if dashed else "-", color=color,
                markeredgecolor="white", label=LABEL.get(mode, mode))
    if flat_note:
        ax.text(0.99, 0.97, "Reactive: one measured cell, drawn flat\n"
                "(token mode caps it by internal_budget, not pressure)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555555")

    # Best single-system baseline + agile's margin over it. Only where BOTH
    # baselines exist at that x: with planning missing, "max" is just reactive,
    # and the margin would overstate agile's advantage.
    have = {m: {x: y for x, y, _, _ in series.get(m, [])} for m in ("reactive", "planning")}
    base = {x: max(have["reactive"][x], have["planning"][x])
            for x in have["reactive"] if x in have["planning"]}
    missing = [x for x in all_x if x not in base]
    if missing and (have["reactive"] or have["planning"]):
        names = ", ".join(LOAD_NAME.get(x, x) if key == "load"
                          else (f"{x // 1024}K" if x >= 1024 else str(x)) for x in missing)
        ax.text(0.01, 0.01, f"no margin shown at {names}: planning/reactive not both run",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color="#999999")
    if base:
        bx = sorted(base, key=lambda x: pos[x])
        bxi = [pos[x] for x in bx]
        for seg in _runs(bxi):
            ax.plot([bxi[i] for i in seg], [base[bx[i]] for i in seg], ls=":", lw=2.4,
                    color=MAXLINE, zorder=2, marker="." if len(seg) == 1 else None)
        ax.plot([], [], ls=":", lw=2.4, color=MAXLINE, label="Max of single-system methods")
        agile = {x: y for x, y, _, _ in series.get("agile", [])}
        for x in bx:
            if x in agile:
                gap = agile[x] - base[x]
                i = pos[x]
                ax.annotate("", xy=(i, agile[x]), xytext=(i, base[x]),
                            arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
                ax.text(i + 0.05, (agile[x] + base[x]) / 2, f"{gap * 100:+.1f}%",
                        fontsize=9.5, va="center", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_xticks(range(len(all_x)))
    ax.set_xticklabels([LOAD_NAME.get(x, x) if key == "load"
                        else (f"{x // 1024}K" if x >= 1024 else str(x))
                        for x in all_x])
    ax.set_xlim(-0.35, max(len(all_x) - 0.65, 0.35))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("SCORE")
    # Snake and Overcooked sit near the floor at 8B: on a 0-1 axis every line is
    # flat and the comparison is invisible. Rescale when the data is all low, and
    # say so on the panel so nobody reads it as a 0-1 plot.
    top = max((p[1] + p[2]) for pts in series.values() for p in pts) if series else 1.0
    if top < 0.25:
        ax.set_ylim(-0.005, top * 1.35)
        ax.text(0.99, 0.97, "y-axis rescaled (paper uses 0-1)", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#C0392B")
    else:
        ax.set_ylim(-0.02, 1.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", default="logs")
    ap.add_argument("--out", default="analysis/figures")
    ap.add_argument("--at-pressure", dest="at_pressure", type=int, default=None,
                    help="pressure to hold fixed for the cognitive-load panel")
    ap.add_argument("--at-load", dest="at_load", default=None,
                    help="load (E/M/H) to hold fixed for the time-pressure panel")
    ap.add_argument("--no-reactive-flat", dest="no_reactive_flat", action="store_true",
                    help="do not extend a single reactive cell across all pressures")
    ap.add_argument("--default-model", dest="default_model", default="qwen3-8b",
                    help="model tag for runs that have no model.txt (the first Qwen batch)")
    a = ap.parse_args()

    rows = load_runs(a.logs_root, a.default_model)
    if not rows:
        raise SystemExit(f"No completed runs with results under {a.logs_root}")
    os.makedirs(a.out, exist_ok=True)

    # group by (model, game): scores use per-game Rmin/Rmax, so mixing games in
    # one figure would average incomparable numbers
    groups = sorted({(r["model"], r["game"]) for r in rows})
    lines = ["model,game,mode,load,time_pressure,n,reward_mean,reward_std,score_mean,score_std"]
    for model, game in groups:
        mrows = [r for r in rows if r["model"] == model and r["game"] == game]

        # ---- summary table ----
        print(f"\n=== {model} · {game} ===")
        print(f"{'mode':<10} {'load':<6} {'pressure':>9} {'n':>3} {'reward':>16} {'score':>16}")
        print("-" * 68)
        buckets = {}
        for r in mrows:
            buckets.setdefault((r["mode"], r["load"], r["pressure"]), []).append(r)
        for (mode, load, pressure), rs in sorted(buckets.items()):
            rw = np.array([x["reward"] for x in rs])
            sc = np.array([x["score"] for x in rs])
            print(f"{mode:<10} {load:<6} {pressure:>9} {len(rs):>3} "
                  f"{rw.mean():>8.2f} ± {rw.std():<5.2f} {sc.mean():>8.3f} ± {sc.std():<5.3f}")
            lines.append(f"{model},{game},{mode},{load},{pressure},{len(rs)},"
                         f"{rw.mean():.4f},{rw.std():.4f},{sc.mean():.4f},{sc.std():.4f}")

        # ---- figure: hold one axis fixed while sweeping the other ----
        fix_press = a.at_pressure or pick_slice(mrows, "load", "pressure")
        fix_load = a.at_load or pick_slice(mrows, "pressure", "load")
        by_load = agg(mrows, "load", "pressure", fix_press)
        by_press = agg(mrows, "pressure", "load", fix_load)
        n_load = max([len(v) for v in by_load.values()] or [0])
        n_press = max([len(v) for v in by_press.values()] or [0])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        _plot(axes[0], by_load, "load", "COGNITIVE LOAD (LOW → HIGH)")
        _plot(axes[1], by_press, "pressure", "TIME PRESSURE (LOW → HIGH)",
              reactive_flat=not a.no_reactive_flat)
        fig.suptitle(f"{game.capitalize()} — {model}", fontsize=13, y=1.02)
        axes[0].set_title(f"score vs cognitive load (time pressure fixed at {fix_press})", fontsize=10.5)
        axes[1].set_title(f"score vs time pressure (load fixed at {LOAD_NAME.get(fix_load, fix_load)})",
                          fontsize=10.5)
        if any(getattr(ax, "_bridged", False) for ax in axes):
            fig.text(0.5, -0.02, "Faded dashed segments join measured points across cells that have not been run.",
                     ha="center", fontsize=8.5, color="#666666")
        if n_load < 2 or n_press < 2:
            fig.text(0.5, 0.005,
                     f"NOTE: {n_load} load point(s), {n_press} pressure point(s). "
                     "A trend needs >=2 per axis — run more cells.",
                     ha="center", fontsize=9, color="#B00020")
        handles = {}
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        order = ["AgileThinker", "Max of single-system methods"]
        keys = [k for k in order if k in handles] + sorted(k for k in handles if k not in order)
        fig.legend([handles[k] for k in keys], keys, loc="lower center",
                   ncol=min(len(keys), 4), frameon=False, bbox_to_anchor=(0.5, -0.10))
        fig.tight_layout()
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{model}_{game}")
        path = os.path.join(a.out, f"score_curves_{slug}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure : {path}")

    with open(os.path.join(a.out, "summary.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary: {os.path.join(a.out, 'summary.csv')}")


if __name__ == "__main__":
    main()
