#!/usr/bin/env python
"""
Build SFT data from the SALT-NLP/RealtimeGym dataset of released trajectories.

Provenance: the dataset card does not name the model that generated the traces,
and the records carry no model field. The repo's reference config
(configs/example-deepseek-v3.2-planning.yaml) points at `deepseek-reasoner`
with tokenizer deepseek-ai/DeepSeek-R1, and the traces are stylistically
uniform, so these are very likely DeepSeek-reasoner outputs -- but that is an
inference, not a documented fact. Worth confirming with the authors before any
claim rests on it.

The dataset stores, per env step: `original_state` (the state block the model was
shown), `thinking` (its reasoning), `action` (what it did) and a structured
`state`. The eval prompt is FAST_AGENT_PROMPT + that state block, so examples
built here are byte-identical to what the model sees at eval time.

Two targets, because they teach different things:

  --style full    prompt -> reasoning + \\boxed{action}
                  Teaches DeepSeek-like play. Its traces are LONG, so a model
                  trained on them is better but SLOWER -- under a token clock
                  that can lower the score. Follow with RL under a length
                  penalty, or evaluate at a generous budget.

  --style short   prompt -> \\boxed{action} only
                  Teaches the policy at a length you can actually afford each
                  step. Cheaper, and a clean measurement of how much the
                  reasoning text is worth.

Usage:
  python analysis/build_sft.py --game freeway --mode reactive --out sft_freeway.jsonl
  python analysis/build_sft.py --game freeway --mode reactive --style short --loads easy medium
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
# one definition, shared with the harness, so train and eval prompts cannot drift
from realtimegym.budget import budget_line  # noqa: E402
HF = "https://huggingface.co/datasets/SALT-NLP/RealtimeGym/resolve/main"
LOADS = ["easy", "medium", "hard"]
PRESSURES = ["4k", "8k", "16k", "32k"]


def templates(game):
    """Read the block scalars out of prompts/<game>.yaml without needing pyyaml."""
    text = (REPO / "src" / "realtimegym" / "prompts" / f"{game}.yaml").read_text()
    out, key, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^([a-z_]+):\s*\|\s*$", line)
        if m:
            if key:
                out[key] = "\n".join(buf)
            key, buf = m.group(1), []
        elif key is not None:
            buf.append(line[2:] if line.startswith("  ") else line)
    if key:
        out[key] = "\n".join(buf)
    return out


ANYTIME_LEAD = ("**Provisional answer** (before reasoning, in case I am cut off):\n"
                "$$\n\\boxed{%s}\n$$\n\n"
                "Now verifying.\n\n")


def anytime_body(act, think):
    """Commit to an answer FIRST, then reason, then restate it.

    The harness reads the LAST \\boxed{} in the output (base.py extract_boxed
    uses rfind), so a generation cut off mid-reasoning leaves the opening answer
    standing. That is the point: at 512 tokens the measured policy collapses to
    pressing "U" on 3200 of 3200 steps, because truncation leaves it with no
    judgement at all. An answer stated up front survives the cut.

    The model therefore has to predict its own conclusion before working it out.
    That is a real skill, and the same one the 9-token policy learns -- this
    format just packs the fast policy and the slow one into a single model.
    """
    tail = think if re.search(r"\\boxed\{\s*" + re.escape(act) + r"\s*\}", think[-400:]) \
        else f"{think}\n\n$$\n\\boxed{{{act}}}\n$$"
    return ANYTIME_LEAD % act + tail

def fetch(name, cache):
    p = cache / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{HF}/{name}", p)
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="freeway", choices=["freeway", "snake", "overcooked"])
    ap.add_argument("--mode", default="reactive", choices=["reactive", "planning", "agile"])
    ap.add_argument("--loads", nargs="*", default=LOADS)
    ap.add_argument("--pressures", nargs="*", default=PRESSURES)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(8)))
    ap.add_argument("--style", default="full", choices=["full", "short", "anytime"])
    ap.add_argument("--budgets", type=int, nargs="*", default=None,
                    help="budget-condition the prompts: one example per budget. "
                         "Needs the harness run with RTG_ANNOUNCE_BUDGET=1.")
    ap.add_argument("--tight-below", type=int, default=2048,
                    help="budgets under this get the answer-only target")
    ap.add_argument("--max-thinking-chars", type=int, default=0,
                    help="drop examples whose reasoning is longer than this (0 = keep all)")
    ap.add_argument("--prompts-only", action="store_true",
                    help="emit prompts without labels, as an RL prompt set")
    ap.add_argument("--cache", default=str(REPO / "rtg_hf_cache"))
    ap.add_argument("--out", default="sft.jsonl")
    a = ap.parse_args()

    tpl = templates(a.game)
    header = tpl["fast_agent_prompt"] if a.mode == "reactive" else tpl["slow_agent_prompt"] + tpl["action_format_prompt"]

    cache = Path(a.cache)
    rows, skipped, missing, lens = [], 0, 0, []
    for load in a.loads:
        for pressure in a.pressures:
            for seed in a.seeds:
                name = f"{a.game}_{load}_{pressure}_seed{seed}_{a.mode}.json"
                try:
                    steps = fetch(name, cache)
                except Exception:
                    missing += 1
                    continue
                for s in steps:
                    think = (s.get("thinking") or "").strip()
                    act = (s.get("action") or "").strip()
                    state = (s.get("original_state") or "").strip()
                    # a few records carry a whole action SEQUENCE ("UDUDSDDS...")
                    # instead of one move -- planning output that leaked into a
                    # reactive file. One step = one action.
                    valid = {"freeway": "UDS", "snake": "LRUD", "overcooked": "UDLRSI"}[a.game]
                    if not act or not state or len(act) != 1 or act not in valid:
                        skipped += 1
                        continue
                    if a.max_thinking_chars and len(think) > a.max_thinking_chars:
                        skipped += 1
                        continue
                    prompt = header + state
                    # Budget conditioning: the same state at several budgets,
                    # each paired with the longest target that budget can
                    # actually finish. Pairing a tight budget with a long
                    # target would teach the model to overrun, not to allocate.
                    for budget in (a.budgets or [None]):
                        p_b = prompt if budget is None else prompt + budget_line(budget)
                        rec = {"prompt": p_b,
                               "game": a.game,
                               "mode": a.mode,
                               "load": load,
                               "pressure": pressure,
                               "seed": seed,
                               "step": s.get("step"),
                               "action": act,
                               "budget": budget}
                        if not a.prompts_only:
                            # the trace usually ends with its own \boxed{...};
                            # only add one when it is missing
                            tight = budget is not None and budget < a.tight_below
                            if a.style == "short" or tight:
                                body = f"$$\n\\boxed{{{act}}}\n$$"
                            elif a.style == "anytime":
                                body = anytime_body(act, think)
                            elif re.search(r"\\boxed\{\s*" + re.escape(act) + r"\s*\}", think[-400:]):
                                body = think
                            else:
                                body = f"{think}\n\n$$\n\\boxed{{{act}}}\n$$"
                            rec["messages"] = [{"role": "user", "content": p_b},
                                               {"role": "assistant", "content": body}]
                            lens.append(len(body))
                        rows.append(rec)

    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} examples -> {a.out}")
    print(f"  files missing: {missing}  steps skipped: {skipped}")
    if lens:
        lens.sort()
        approx = lambda c: c / 4          # ~4 chars per token, rough
        print(f"  completion chars: median {lens[len(lens)//2]}  p90 {lens[int(.9*len(lens))]}  max {lens[-1]}")
        print(f"  ~tokens:          median {approx(lens[len(lens)//2]):.0f}  p90 {approx(lens[int(.9*len(lens))]):.0f}")
    acts = {}
    for r in rows:
        acts[r["action"]] = acts.get(r["action"], 0) + 1
    print(f"  action balance: {dict(sorted(acts.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
