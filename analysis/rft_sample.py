#!/usr/bin/env python
"""
Rejection sampling for Freeway: sample G completions per state from the SFT
model, score each by forking the env, keep the winner, write SFT JSONL.

This is the cheap half of "RL". It optimises the same objective a policy
gradient would -- pick actions that stay survivable, and prefer short ones --
but by filtering samples instead of differentiating through them, so it reuses
the serving and training code that already works.

A state is only kept when the sampling actually found something better than
what the model does on its own. Training on states where every sample ties
just reinforces current behaviour and dilutes the signal.

Usage (inside the CHTC job, with vLLM already serving the adapter):
  python analysis/rft_sample.py --url http://127.0.0.1:8000/v1 --model sft \\
      --runs 'rtg_results/logs/sft_freeway_full_none_*_4096' --out rft.jsonl
"""
import argparse
import concurrent.futures as cf
import collections
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from rl_reward import score, parse_action          # noqa: E402
from rl_states import collect_csv, load_of         # noqa: E402
import glob                                        # noqa: E402


_TOK = None


def ntokens(text):
    """Per-choice token count. usage.completion_tokens is the total across all
    n choices, so it cannot price an individual sample -- and pricing one is
    the whole point of the length term."""
    global _TOK
    if _TOK is None:
        return max(1, round(len(text) / 3.4))       # fallback: measured ratio
    return len(_TOK(text, add_special_tokens=False)["input_ids"])


def generate(url, model, prompt, n, max_tokens, temperature, timeout=600):
    """vLLM's OpenAI endpoint. enable_thinking=false matches the reactive
    config exactly -- a different chat template would make these samples
    unusable as training data for that same config."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "n": n, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return [((c["message"].get("content") or ""), None) for c in out["choices"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="sft")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="rft.jsonl")
    ap.add_argument("--group", type=int, default=8, help="samples per state")
    ap.add_argument("--states", type=int, default=1000)
    ap.add_argument("--mid-lane-frac", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.3,
                    help="length penalty, in reward units per full budget")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--keep-ties", action="store_true",
                    help="also keep states where no sample beat the model")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B",
                    help="only for counting sample length; '' falls back to chars/3.4")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    global _TOK
    if a.tokenizer:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
        except Exception as e:
            print(f"  (tokenizer unavailable, using chars/3.4: {repr(e)[:70]})", flush=True)

    import random
    rng = random.Random(a.seed)

    # --- states -------------------------------------------------------------
    recs = []
    for pat in a.runs:
        for d in sorted(glob.glob(pat)):
            if not os.path.isdir(d):
                continue
            for f in sorted(glob.glob(os.path.join(d, "*", "*_*.csv"))):
                got, _ = collect_csv(f, load_of(d))
                recs += got
    mid = [r for r in recs if 2 <= r["pos"] <= 7]
    rest = [r for r in recs if not (2 <= r["pos"] <= 7)]
    n_mid = min(len(mid), int(a.states * a.mid_lane_frac))
    recs = rng.sample(mid, n_mid) + rng.sample(rest, min(len(rest), a.states - n_mid))
    rng.shuffle(recs)
    print(f"states: {len(recs)} ({n_mid} mid-lane)  group={a.group}  lam={a.lam}", flush=True)

    # --- sample + score -----------------------------------------------------
    t0 = time.time()
    rows, stats = [], collections.Counter()
    deltas, kept_tok, base_tok = [], [], []

    def work(r):
        try:
            outs = generate(a.url, a.model, r["prompt"], a.group, a.max_tokens, a.temperature)
        except (urllib.error.URLError, OSError, KeyError) as e:
            return r, None, repr(e)[:90]
        return r, [(c, ntokens(c)) for c, _ in outs], None

    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for i, (r, outs, err) in enumerate(ex.map(work, recs)):
            if err:
                stats["gen_error"] += 1
                if stats["gen_error"] <= 3:
                    print(f"  gen error: {err}", flush=True)
                continue
            scored = [score(r["env"], c, t, a.max_tokens, a.lam) for c, t in outs]
            rewards = [s[0] for s in scored]
            bi = max(range(len(rewards)), key=lambda j: (rewards[j], -outs[j][1]))
            # what the model does unprompted is its logged greedy action
            base_r, _ = score(r["env"], f"\\boxed{{{r['logged_action']}}}",
                              r["logged_tokens"], a.max_tokens, a.lam)
            stats["states"] += 1
            stats[f"best_{scored[bi][1]['action']}"] += 1
            if scored[bi][1]["collided"]: stats["best_still_collides"] += 1
            if rewards[bi] > base_r + 1e-9:
                stats["improved"] += 1
                deltas.append(rewards[bi] - base_r)
            elif not a.keep_ties:
                stats["skipped_tie"] += 1
                continue
            kept_tok.append(outs[bi][1]); base_tok.append(r["logged_tokens"] or 0)
            rows.append({"messages": [{"role": "user", "content": r["prompt"]},
                                      {"role": "assistant", "content": outs[bi][0]}],
                         "action": scored[bi][1]["action"], "reward": rewards[bi],
                         "base_reward": base_r, "tokens": outs[bi][1],
                         "pos": r["pos"], "load": r["load"], "seed": r["seed"]})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(recs)}  kept={len(rows)}  "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)

    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    n = max(1, stats["states"])
    print(f"\nwrote {len(rows)} examples -> {a.out}   ({(time.time()-t0)/60:.1f} min)")
    print(f"  states sampled            {stats['states']}")
    print(f"  improved on the model     {stats['improved']} ({100*stats['improved']/n:.1f}%)")
    print(f"  skipped (no improvement)  {stats['skipped_tie']}")
    print(f"  best sample still collides{stats['best_still_collides']:>5}")
    print(f"  generation errors         {stats['gen_error']}")
    if deltas:
        print(f"  mean reward gain          {sum(deltas)/len(deltas):+.2f}")
    if kept_tok and any(base_tok):
        bt = [b for b in base_tok if b]
        print(f"  tokens: kept median {sorted(kept_tok)[len(kept_tok)//2]}  "
              f"vs model median {sorted(bt)[len(bt)//2]}")
    print(f"  chosen action mix         "
          f"{ {k[5:]: v for k, v in stats.items() if k.startswith('best_') and k != 'best_still_collides'} }")


if __name__ == "__main__":
    main()
