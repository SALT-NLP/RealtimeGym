# Baseline results

Per-run `args.log` (config + one line per seed with its reward) and the figures
built from them. The per-step CSVs and the CHTC tarballs are **not** committed —
they are ~900 MB, and everything here can be rebuilt without them.

```
runs/<runtag>/<game>_<load>_<pressure>_<mode>_<ibudget>_<timestamp>/args.log
runs/<runtag>/model.txt        # which models produced the run
figures/                       # score curves per (model, game) + summary.csv
```

Regenerate the figures from this directory:

```bash
python3 analysis/plot_baseline.py --logs_root results/runs --out results/figures
```

## What is here

| model | game | cells |
| --- | --- | --- |
| Qwen3-8B (thinking on/off) | Freeway | load panel E/M/H + pressure sweep 4K-32K |
| Qwen3-8B | Snake | full 3x3 grid, 32 runs per cell |
| Qwen3-8B | Overcooked | 3 modes at Medium |
| Llama-3.1-8B + R1-Distill-Llama-8B | Freeway | load panel + pressure sweep |
| Llama-3.1-8B + R1-Distill-Llama-8B | Snake | 7 of 9 cells, 32 runs per cell |

Scores use the paper's Table 4 bounds: Freeway (0, 89), Snake (-1, 15),
Overcooked (0, 56). Runs are 8 game seeds x 1 sample unless `n` says otherwise;
the Snake cells are 8 seeds x 4 samples.

Caveats worth carrying: outcomes are close to all-or-nothing, so per-cell
standard deviations are large, and two identical Freeway/Easy runs differed by
0.115 from sampling alone. Only the ordering across cells is solid. Snake and
Overcooked sit near the floor for every agent design at 8B.

The jobs that produced these live in `Badger-RL/llm-starter` under
`chtc/realtimegym/`.
