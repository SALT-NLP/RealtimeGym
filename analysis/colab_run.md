# RealtimeGym on Colab (A100) — one grid cell

Mirrors the CHTC job. Pick a cell that is still IDLE on CHTC and `condor_rm` it
first so you are not running the same thing twice.

Runtime > Change runtime type > **A100 GPU** (and High-RAM if offered).

---
### Cell 1 — verify the GPU
```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```
Must show ~40GB. If it shows a T4, stop: 8B bf16 will not fit and a quantized
run is not comparable to the CHTC numbers.

---
### Cell 2 — install (~5-8 min; may ask you to restart the runtime)
```python
!pip install -q vllm
!pip install -q pygame gym gym_notices python-dotenv
# IPython/imageio already present on Colab
```
If prompted to restart, restart then skip straight to Cell 3.

---
### Cell 3 — get the code + write configs
```python
!git clone -q https://github.com/nabanitabag/RealtimeGym.git /content/RealtimeGym
%cd /content/RealtimeGym
!git checkout -q main

planning = """model: Qwen/Qwen3-8B
url: http://127.0.0.1:8000/v1
api_key: dummy
inference_parameters:
  temperature: 0.6
  top_p: 0.95
  max_tokens: 16384
  extra_body:
    chat_template_kwargs:
      enable_thinking: true
tokenizer: Qwen/Qwen3-8B
"""
reactive = """model: Qwen/Qwen3-8B
url: http://127.0.0.1:8000/v1
api_key: dummy
inference_parameters:
  temperature: 0.7
  top_p: 0.8
  max_tokens: 2048
  extra_body:
    chat_template_kwargs:
      enable_thinking: false
"""
open("configs/chtc-qwen3-8b-planning.yaml", "w").write(planning)
open("configs/chtc-qwen3-8b-reactive.yaml", "w").write(reactive)
print("configs written")
```

---
### Cell 4 — start vLLM in the background and wait
```python
import subprocess, urllib.request, time, os
os.environ["VLLM_USAGE_DISABLE"] = "1"
log = open("/content/vllm.log", "w")
subprocess.Popen(
    ["vllm", "serve", "Qwen/Qwen3-8B", "--port", "8000",
     "--max-model-len", "40960", "--gpu-memory-utilization", "0.90"],
    stdout=log, stderr=subprocess.STDOUT)

for i in range(180):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
        print(f"vLLM healthy after {i*10}s"); break
    except Exception:
        time.sleep(10)
else:
    print(open("/content/vllm.log").read()[-4000:])
    raise SystemExit("vLLM never came up")
```

---
### Cell 5 — sanity check (trace must be visible)
```python
import json, urllib.request
b = json.dumps({"model": "Qwen/Qwen3-8B",
                "messages": [{"role": "user", "content": "What is 17*23?"}],
                "max_tokens": 300}).encode()
r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                           data=b, headers={"Content-Type": "application/json"})
m = json.load(urllib.request.urlopen(r, timeout=180))["choices"][0]["message"]
print("reasoning_content:", (m.get("reasoning_content") or "MISSING")[:150])
print("content:", (m.get("content") or "")[:150])
```
PASS if `reasoning_content` has text **or** `content` starts with `<think>`.

---
### Cell 6 — run the cell (edit MODE / LOAD / IBUDGET)
```python
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONPATH"] = "/content/RealtimeGym/src"

MODE, LOAD, PRESSURE = "agile", "H", 4096
IBUDGET = 0 if MODE == "planning" else (4096 if MODE == "reactive" else 2048)

!python -m realtimegym.agile_eval \
  --game freeway --cognitive_load {LOAD} --mode {MODE} \
  --time_pressure {PRESSURE} --internal_budget {IBUDGET} \
  --planning-model-config configs/chtc-qwen3-8b-planning.yaml \
  --reactive-model-config configs/chtc-qwen3-8b-reactive.yaml \
  --seed_num 8 --repeat_times 1 --log_dir logs/colab
```

---
### Cell 7 — save results before the session dies
```python
!tar -czf /content/results_colab.tar.gz -C /content/RealtimeGym logs/colab
from google.colab import files; files.download("/content/results_colab.tar.gz")
```
Untar it next to the CHTC results; `plot_baseline.py` reads it the same way.
