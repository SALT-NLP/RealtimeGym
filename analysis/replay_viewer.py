#!/usr/bin/env python
"""
Replay finished Freeway runs into a self-contained HTML viewer -- no LLM calls.

Each per-seed CSV logs, for every env step, the board *before* the action
(`render`), the `action`, the `reward`, and what the agent was thinking:
`plan` (the planning thread's text revealed so far) and `model1_response`
(the reactive thread's answer). Freeway is deterministic given its seed, so we
rebuild the game by replaying the logged actions through the real env and check
every step against the logged board.

The board is drawn in the browser from per-step car/chicken positions using the
repo's own sprites (same layout as FreewayRender), so a run with 8 seeds stays a
few MB instead of tens of MB of pre-rendered frames.

Usage:
  python analysis/replay_viewer.py --run_dir rtg_results/logs/vanilla/freeway_M_4096_agile_2048_*
  python analysis/replay_viewer.py --run_dir <dir> --seeds 0 3 --out viewer.html

Open the resulting .html in a browser.
"""
import argparse
import base64
import glob
import html
import json
import math
import os
import re
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import pandas as pd  # noqa: E402

import realtimegym  # noqa: E402

LOAD_VERSION = {"E": 0, "M": 1, "H": 2}
PLAN_T1 = re.compile(r"t_1\s*=\s*(\d+)")
ASSETS = os.path.join(REPO, "src", "realtimegym", "environments", "render", "assets", "freeway")
SPRITES = {"chicken": "chicken.png", "car_1": "car1.png", "car_2": "car2.png",
           "car_3": "car3.png", "car_4": "car4.png", "grey": "grey.png",
           "yellow": "yellow.png", "grass": "grass.png", "target": "map-pin.png"}


def read_args_log(run_dir):
    cfg = {}
    with open(os.path.join(run_dir, "args.log")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("seed:") or set(line) == {"-"}:
                continue
            k, _, v = line.partition(":")
            cfg[k.strip()] = v.strip()
    return cfg


def text(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def norm_board(s):
    return "\n".join(line.rstrip() for line in str(s).strip("\n").splitlines())


def snapshot(env):
    """Everything FreewayRender.render() reads: cars, chicken row, turn, terminal."""
    cars = [[c[0], c[1], c[3], c[4]] for c in env.cars
            if c[0] is not None and c[3] is not None]
    return {"pos": env.pos, "turn": env.game_turn, "end": bool(env.terminal), "cars": cars}


def replay_seed(csv_path, load):
    seed_idx = int(os.path.basename(csv_path).split("_")[1].split(".")[0])
    df = pd.read_csv(csv_path)
    env, real_seed, _ = realtimegym.make(f"Freeway-v{LOAD_VERSION[load]}", seed=seed_idx)
    obs, done = env.reset()

    plans, plan_index = [], {}   # dedupe growing plan prefixes per grounding turn
    steps, boards, mismatch = [], [], None
    gen_t1 = None                # planning mode: turn the current generation started
    for i, row in df.iterrows():
        logged = norm_board(row.get("render", ""))
        if mismatch is None and logged and norm_board(obs["state_string"]) != logged:
            mismatch = i
        boards.append(snapshot(env))

        # Agile logs the revealed plan in `plan` with a "t_1 = N" header.
        # Planning mode leaves `plan` for its action queue and puts the growing
        # reasoning in `model2_response`; a non-empty `model2_prompt` marks the
        # step where a new generation was grounded.
        plan = text(row.get("plan"))
        queue = ""
        if "Guidance from a Previous Thinking Model" not in plan:
            queue, plan = plan, text(row.get("model2_response"))
            if text(row.get("model2_prompt")):
                gen_t1 = env.game_turn
            if plan:
                plan = f"t_1 = {gen_t1}\n{plan}" if gen_t1 is not None else "\n" + plan
        plan_id, plan_len, t1 = None, 0, None
        if plan:
            m = PLAN_T1.search(plan[:200])
            t1 = int(m.group(1)) if m else None
            body = plan.split("\n", 1)[1] if "\n" in plan else plan
            key = t1 if t1 is not None else f"x{i}"
            j = plan_index.get(key)
            if j is not None and (body.startswith(plans[j]) or plans[j].startswith(body)):
                if len(body) > len(plans[j]):
                    plans[j] = body
            else:
                j = len(plans)
                plans.append(body)
                plan_index[key] = j
            plan_id, plan_len = j, len(body)

        action = text(row.get("action")) or "S"
        turn = env.game_turn
        obs, done, reward, collided = env.step(action)
        steps.append({
            "turn": turn, "action": action, "reward": int(reward), "collided": bool(collided),
            "plan": plan_id, "planLen": plan_len, "t1": t1, "queue": queue,
            "reactive": text(row.get("model1_response")),
            "tok1": text(row.get("model1_token_num")),
        })
        if done:
            break
    boards.append(snapshot(env))   # final board
    return {
        "seedIdx": seed_idx, "envSeed": int(real_seed),
        "steps": steps, "boards": boards, "plans": plans,
        "finalReward": int(env.reward),
        "success": bool(env.terminal and env.game_turn < 100),
        "turns": env.game_turn, "mismatch": mismatch,
    }


PAGE = r"""<title>__TITLE__</title>
<style>
:root{--bg:#f6f6f3;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e2dc;--accent:#0f7b5f;
--bad:#b3261e;--warn:#9a6700;--chip:#efefea;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--panel:#1e1e1c;--ink:#ecece6;
--muted:#a3a39b;--line:#33332f;--accent:#4fc59d;--bad:#ff8a80;--warn:#e3b341;--chip:#2a2a27}}
:root[data-theme="dark"]{--bg:#151514;--panel:#1e1e1c;--ink:#ecece6;--muted:#a3a39b;--line:#33332f;
--accent:#4fc59d;--bad:#ff8a80;--warn:#e3b341;--chip:#2a2a27}
body{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif;margin:0;padding:16px}
.wrap{max-width:1280px;margin:0 auto}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--muted);margin:0 0 14px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px}
select,button{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer}
button:hover,select:hover{border-color:var(--muted)}
input[type=range]{flex:1;min-width:160px;accent-color:var(--accent)}
.grid{display:grid;grid-template-columns:minmax(260px,420px) 1fr;gap:14px}
@media (max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;min-width:0}
canvas{width:100%;height:auto;border-radius:8px;display:block;image-rendering:auto}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 0;align-items:center}
.chip{background:var(--chip);border-radius:999px;padding:3px 10px;font-variant-numeric:tabular-nums}
.chip b{font-weight:650}.chip.bad{background:var(--bad);color:#fff}.chip.ok{background:var(--accent);color:#fff}
.stale{color:var(--warn);font-weight:650}
.act{font-size:22px;font-weight:700}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 6px;display:flex;flex-wrap:wrap;justify-content:space-between;gap:4px 8px}
h2 .meta{text-transform:none;letter-spacing:0;font-size:12px}
.txt{font-family:var(--mono);font-size:12px;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow:auto;
border-top:1px solid var(--line);padding-top:8px;margin:0}
.hidden-part{color:var(--muted);opacity:.4}
.note{color:var(--muted);font-size:12px;margin:8px 0 0}
.strip{display:flex;gap:1px;margin:10px 0 0;height:14px}
.strip span{flex:1;border-radius:2px;background:var(--chip);cursor:pointer;min-width:2px}
.strip span.U{background:var(--accent)}.strip span.D{background:var(--warn)}.strip span.hit{background:var(--bad)}
.strip span.cur{outline:2px solid var(--ink);outline-offset:1px}
.alert{background:var(--bad);color:#fff;border-radius:8px;padding:6px 10px;margin-bottom:10px}
</style>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="sub">__SUBTITLE__</p>
  <div class="bar">
    <select id="seed" aria-label="Seed"></select>
    <button id="prev" aria-label="Previous step">◀</button>
    <button id="play">Play</button>
    <button id="next" aria-label="Next step">▶</button>
    <select id="speed" aria-label="Speed"><option value="900">slow</option><option value="450" selected>normal</option><option value="150">fast</option></select>
    <input id="slider" type="range" min="0" value="0" aria-label="Step">
    <span id="stepLbl" class="chip"></span>
  </div>
  <div id="alert"></div>
  <div class="grid">
    <div class="card">
      <canvas id="board" width="540" height="600" aria-label="Freeway board"></canvas>
      <div class="chips" id="chips"></div>
      <div class="strip" id="strip"></div>
      <p class="note">Board is shown <b>before</b> the action at this turn. The chicken starts at the bottom and must reach the pin at the top. Strip: green = Up, amber = Down, grey = Stay, red = hit by a car. Keys: ← → space.</p>
    </div>
    <div style="display:grid;gap:14px;min-width:0;align-content:start">
      <div class="card"><h2><span id="planTitle">Planning thread — what the reactive model could see</span><span class="meta" id="planMeta"></span></h2><pre class="txt" id="plan"></pre>
        <p class="note">Dark text was revealed by this turn; faded text is still being "thought" and was not visible yet.</p></div>
      <div class="card"><h2><span>Reactive thread — the decision</span><span class="meta" id="reactMeta"></span></h2><pre class="txt" id="react"></pre></div>
    </div>
  </div>
</div>
<script>
const RUNS = __DATA__;
const SPRITES = __SPRITES__;
const CS = 60, W = 9 * CS, H = 10 * CS;
const $ = id => document.getElementById(id);
const img = {};
let pending = Object.keys(SPRITES).length;
for (const [k, b64] of Object.entries(SPRITES)) {
  const im = new Image();
  im.onload = () => { if (--pending === 0) show(); };
  im.src = "data:image/png;base64," + b64;
  img[k] = im;
}
const ctx = $("board").getContext("2d");

// Mirrors src/realtimegym/environments/render/freeway_render.py
function drawBoard(b) {
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
  if (pending) return;
  for (let i = 0; i < 10; i++) for (let j = 0; j < 9; j++) {
    const x = j * CS, y = i * CS;
    if (i === 0 || i === 9) ctx.drawImage(img.grass, x, y, CS * .95, CS * .95);
    else {
      ctx.drawImage(j === 4 ? img.yellow : img.grey, x, y, CS * .95, CS * .95);
      if (j < 8) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.beginPath();
        ctx.moveTo((j + 1) * CS - .5, y); ctx.lineTo((j + 1) * CS - .5, y + CS); ctx.stroke(); }
    }
  }
  for (const [x, y, speed, len] of b.cars) {
    const right = speed > 0, cy = y * CS;
    const cx = right ? (x - len + 1) * CS : x * CS;
    if (cx + len * CS <= 0 || cx >= W) continue;
    const sp = img["car_" + len];
    if (right) ctx.drawImage(sp, cx, cy, len * CS, CS);
    else { ctx.save(); ctx.translate(cx + len * CS, cy); ctx.scale(-1, 1); ctx.drawImage(sp, 0, 0, len * CS, CS); ctx.restore(); }
  }
  ctx.drawImage(img.target, 4 * CS, 0, CS * .95, CS * .95);
  ctx.drawImage(img.chicken, 4 * CS, b.pos * CS, CS * .95, CS * .95);
  ctx.font = "bold 24px system-ui"; const t = `Turn: ${b.turn}`, tw = ctx.measureText(t).width;
  ctx.fillStyle = "#000"; ctx.fillRect(CS / 4 - 5, CS / 4 - 4, tw + 10, 32);
  ctx.fillStyle = "#fff"; ctx.textBaseline = "top"; ctx.fillText(t, CS / 4, CS / 4);
  if (b.end) {
    ctx.fillStyle = "rgba(0,0,0,.5)"; ctx.fillRect(0, 0, W, H);
    const ok = b.turn < 100; ctx.font = "bold 36px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = ok ? "#00ff00" : "#ff0000";
    ctx.fillText(ok ? `SUCCESS in ${b.turn} turns!` : "GAME OVER", W / 2, H / 2); ctx.textAlign = "left";
  }
}

let run = RUNS[0], i = 0, timer = null;
RUNS.forEach((r, k) => {
  const o = document.createElement("option");
  o.value = k;
  o.textContent = `seed ${r.seedIdx} — ${r.success ? "crossed in " + r.turns + " turns" : "did not cross"} (reward ${r.finalReward})`;
  $("seed").appendChild(o);
});
const esc = s => s.replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
const arrow = {U: "↑ Up", D: "↓ Down", S: "· Stay"};
const hasPlanner = r => r.steps.some(s => s.plan !== null);

function load(k) {
  run = RUNS[k]; i = 0;
  const reactive = run.steps.some(s => s.reactive);
  $("planTitle").textContent = reactive ? "Planning thread — what the reactive model could see"
                                        : "Planning thread — reasoning so far (acts on its committed plan)";
  buildStrip(); show();
}
function buildStrip() {
  const s = $("strip"); s.innerHTML = "";
  run.steps.forEach((st, k) => {
    const e = document.createElement("span");
    e.className = (st.collided ? "hit " : "") + st.action;
    e.title = `turn ${st.turn}: ${st.action}${st.collided ? " (hit)" : ""}`;
    e.onclick = () => { stop(); i = k; show(); };
    s.appendChild(e);
  });
  $("slider").max = run.steps.length;
  $("alert").innerHTML = run.mismatch === null ? "" :
    `<div class="alert">Replay diverged from the logged board at step ${run.mismatch}; boards after that may not match what the agent saw.</div>`;
}
function show() {
  const n = run.steps.length;
  i = Math.max(0, Math.min(i, n));
  $("slider").value = i;
  drawBoard(run.boards[i]);
  [...$("strip").children].forEach((e, k) => e.classList.toggle("cur", k === i));
  if (i === n) {
    $("stepLbl").textContent = `end · ${n} steps`;
    $("chips").innerHTML = `<span class="chip ${run.success ? "ok" : "bad"}"><b>${run.success ? "Crossed" : "Out of time"}</b></span><span class="chip">final reward <b>${run.finalReward}</b> · score <b>${(run.finalReward / 89).toFixed(3)}</b></span>`;
    ["plan", "react", "planMeta", "reactMeta"].forEach(id => $(id).textContent = "");
    return;
  }
  const st = run.steps[i];
  $("stepLbl").textContent = `step ${i + 1} / ${n}`;
  let chips = `<span class="chip">turn <b>${st.turn}</b></span><span class="chip act">${arrow[st.action] || st.action}</span><span class="chip">reward <b>${st.reward}</b></span>`;
  if (st.collided) chips += `<span class="chip bad"><b>Hit — back to start</b></span>`;
  if (st.queue) chips += `<span class="chip" title="actions left in the committed plan">queued plan <b>${esc(st.queue)}</b></span>`;
  $("chips").innerHTML = chips;

  if (st.plan === null) {
    $("plan").innerHTML = `<span class="hidden-part">${hasPlanner(run) ? "Nothing revealed from the planning thread yet." : "This mode has no planning thread."}</span>`;
    $("planMeta").textContent = "";
  } else {
    const full = run.plans[st.plan], seen = full.slice(0, st.planLen);
    const rest = full.slice(st.planLen, st.planLen + 800);
    $("plan").innerHTML = esc(seen) + `<span class="hidden-part">${esc(rest)}${full.length > st.planLen + 800 ? "…" : ""}</span>`;
    if (st.t1 === null) $("planMeta").textContent = "";
    else {
      const age = st.turn - st.t1;
      $("planMeta").innerHTML = `grounded on turn ${st.t1} · <span class="${age >= 3 ? "stale" : ""}">${age} turn${age === 1 ? "" : "s"} stale</span> · ${Math.round(100 * st.planLen / full.length)}% revealed`;
    }
    const pre = $("plan"); pre.scrollTop = pre.scrollHeight;
    // keep the reveal boundary in view rather than the faded tail
    const boundary = pre.scrollHeight * (seen.length / Math.max(1, seen.length + rest.length));
    pre.scrollTop = Math.max(0, boundary - pre.clientHeight + 40);
  }
  $("react").textContent = st.reactive ||
    (hasPlanner(run) ? "(planning-only agent: no reactive thread. It executes the queued plan, or the default action when none is ready.)"
                     : "(no reactive text logged)");
  $("reactMeta").textContent = st.tok1 ? `${st.tok1} tokens` : "";
  $("react").scrollTop = $("react").scrollHeight;
}
function stop() { clearInterval(timer); timer = null; $("play").textContent = "Play"; }
$("play").onclick = () => {
  if (timer) return stop();
  $("play").textContent = "Pause";
  timer = setInterval(() => { if (i >= run.steps.length) return stop(); i++; show(); }, +$("speed").value);
};
$("speed").onchange = () => { if (timer) { stop(); $("play").click(); } };
$("prev").onclick = () => { stop(); i--; show(); };
$("next").onclick = () => { stop(); i++; show(); };
$("slider").oninput = e => { stop(); i = +e.target.value; show(); };
$("seed").onchange = e => { stop(); load(+e.target.value); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === "ArrowRight") { stop(); i++; show(); }
  else if (e.key === "ArrowLeft") { stop(); i--; show(); }
  else if (e.key === " ") { e.preventDefault(); $("play").click(); }
});
// deep link: replay.html#seed=4&step=10 (seed = seed index, step = 0-based)
const h = new URLSearchParams(location.hash.slice(1));
const k0 = Math.max(0, RUNS.findIndex(r => String(r.seedIdx) === h.get("seed")));
$("seed").value = k0;
load(k0);
if (h.has("step")) { i = +h.get("step"); show(); }
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="directory holding args.log and {r}_{seed}.csv")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="seed indices (default: all)")
    ap.add_argument("--out", default=None, help="output .html (default: <run_dir>/replay.html)")
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    cfg = read_args_log(run_dir)
    if cfg.get("game", "freeway") != "freeway":
        raise SystemExit("Only Freeway replays are supported.")
    load = cfg.get("cognitive_load", "M")
    csvs = sorted(glob.glob(os.path.join(run_dir, "*_*.csv")))
    if a.seeds is not None:
        csvs = [c for c in csvs if int(os.path.basename(c).split("_")[1].split(".")[0]) in a.seeds]
    if not csvs:
        raise SystemExit(f"No per-seed CSVs found in {run_dir}")

    runs = []
    for c in csvs:
        r = replay_seed(c, load)
        flag = "OK" if r["mismatch"] is None else f"DIVERGED at step {r['mismatch']}"
        print(f"  seed {r['seedIdx']}: {len(r['steps'])} steps, reward {r['finalReward']}, replay {flag}")
        runs.append(r)

    sprites = {}
    for key, fn in SPRITES.items():
        with open(os.path.join(ASSETS, fn), "rb") as f:
            sprites[key] = base64.b64encode(f.read()).decode()

    mode = cfg.get("mode", "?")
    title = f"Freeway replay — {mode}, load {load}, {cfg.get('time_pressure', '?')} tokens/step"
    subtitle = (f"internal budget {cfg.get('internal_budget', '?')} · {len(runs)} seed(s) · "
                f"{os.path.basename(run_dir)}")
    page = (PAGE.replace("__TITLE__", html.escape(title))
                .replace("__SUBTITLE__", html.escape(subtitle))
                .replace("__SPRITES__", json.dumps(sprites))
                .replace("__DATA__", json.dumps(runs).replace("</", "<\\/")))
    out = a.out or os.path.join(run_dir, "replay.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"Viewer: {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
