#!/usr/bin/env python
"""
LoRA SFT on RealtimeGym trajectories (output of analysis/build_sft.py).

The targets are the dataset's released reasoning traces -- very likely
DeepSeek-reasoner's, per the repo's reference config, though the dataset card
does not say so (see build_sft.py). This is therefore distillation of another
model's reasoning into the base model, not the base model learning to reason.

Loss is computed on the ASSISTANT TURN ONLY -- the prompt is masked out. Without
that the model spends most of its gradient learning to reproduce the game rules,
which it is already given at eval time.

The adapter is saved so vLLM can serve it directly:
    vllm serve <base> --enable-lora --lora-modules sft=<out_dir>

Usage (inside the CHTC job):
  python analysis/train_sft.py --data sft.jsonl --base Qwen/Qwen3-8B --out /staging/.../sft_adapter
"""
import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model


class Steps(Dataset):
    """One example = one env step. Labels are -100 over the prompt."""

    def __init__(self, path, tok, max_len, balance=None, limit=0, seed=0):
        rows = [json.loads(l) for l in open(path)]
        if balance:
            rows = rebalance(rows, balance, seed)
        if limit:
            random.Random(seed).shuffle(rows)
            rows = rows[:limit]
        self.rows, self.tok, self.max_len = rows, tok, max_len
        # Truncating at max_len would cut the trailing \boxed{} off the longest
        # traces, teaching the model to reason and then never answer. Drop those
        # instead -- a short example is useless, a headless one is harmful.
        keep = [r for r in self.rows if len(self._ids(r)[1]) <= max_len]
        if len(keep) != len(self.rows):
            print(f"  dropped {len(self.rows) - len(keep)} example(s) longer than {max_len} tokens")
            self.rows = keep

    def _ids(self, row):
        m = row["messages"]
        prompt = self.tok.apply_chat_template(
            [m[0]], tokenize=False, add_generation_prompt=True,
            **({"enable_thinking": False} if "qwen3" in self.tok.name_or_path.lower() else {}))
        full = prompt + m[1]["content"] + (self.tok.eos_token or "")
        return (self.tok(prompt, add_special_tokens=False)["input_ids"],
                self.tok(full, add_special_tokens=False)["input_ids"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # prompt exactly as served at eval time, then the target turn
        p_ids, f_ids = self._ids(self.rows[i])
        f_ids = f_ids[: self.max_len]
        labels = list(f_ids)
        for j in range(min(len(p_ids), len(labels))):
            labels[j] = -100
        return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}


def rebalance(rows, mode, seed):
    """The traces drive forward: ~79% U / 16% S / 4% D on Freeway. Training on that
    as-is teaches 'always press U', which is already the reactive agent's failure
    mode. 'cap' downsamples the majority action to 2x the runner-up."""
    by = {}
    for r in rows:
        by.setdefault(r["action"], []).append(r)
    counts = {k: len(v) for k, v in by.items()}
    if mode == "cap" and len(counts) > 1:
        second = sorted(counts.values())[-2]
        keep = max(second * 2, 1)
        rng = random.Random(seed)
        out = []
        for k, v in by.items():
            out += v if len(v) <= keep else rng.sample(v, keep)
        print(f"  rebalanced {counts} -> { {k: min(len(v), keep) for k, v in by.items()} }")
        return out
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--balance", choices=["none", "cap"], default="cap")
    ap.add_argument("--limit", type=int, default=0, help="cap examples (smoke tests)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = Steps(a.data, tok, a.max_len, None if a.balance == "none" else a.balance, a.limit)
    n_val = max(1, int(len(ds) * a.val_frac))
    idx = list(range(len(ds)))
    random.Random(0).shuffle(idx)
    val = torch.utils.data.Subset(ds, idx[:n_val])
    train = torch.utils.data.Subset(ds, idx[n_val:])
    print(f"examples: {len(train)} train / {len(val)} val, max_len {a.max_len}")

    # Prove the masking: only the assistant turn should survive. A silent
    # off-by-one here trains on the game rules and looks like a normal loss curve.
    ex = ds[idx[-1]]
    kept = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    print(f"  loss span: {len(kept)}/{len(ex['input_ids'])} tokens; decoded ->")
    print("  " + repr(tok.decode(kept)[:300]))

    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=None)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum, num_train_epochs=a.epochs,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        # "no": save_pretrained below writes the adapter. Trainer checkpoints
        # also carry optimizer state (~1GB), which then gets copied to /staging
        # for nothing.
        bf16=True, logging_steps=10, save_strategy="no",
        eval_strategy="epoch", per_device_eval_batch_size=a.batch,
        gradient_checkpointing=True, report_to=[], remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    m = trainer.evaluate()
    if "eval_loss" in m:
        print(f"final eval loss {m['eval_loss']:.4f}  (perplexity {math.exp(m['eval_loss']):.2f})")
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"adapter saved to {a.out}")
    print(f"serve with: vllm serve {a.base} --enable-lora --lora-modules sft={a.out} --max-lora-rank {a.rank}")


if __name__ == "__main__":
    main()
