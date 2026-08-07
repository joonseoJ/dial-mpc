"""Live Go2 reward-weight controls for the asynchronous DIAL-MPC planner."""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request


OBJECTIVE_NAMES = ("tracking", "stability", "gait")
SHM_NAME = "dial_reward_weights"
RESET_SHM_NAME = "dial_reset_counter"


def _parse_weights(text: str) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split(",")], dtype=np.float32)
    if values.shape != (3,):
        raise argparse.ArgumentTypeError("expected three comma-separated weights")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise argparse.ArgumentTypeError("weights must be finite and non-negative")
    return values


def _attach(name: str):
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError as exc:
        raise SystemExit(
            "control shared memory is not available; start dial-mpc-sim first"
        ) from exc
    # This process only attaches.  On Python 3.10 the resource tracker would
    # otherwise unlink the simulator-owned segment when a one-shot --get/--set
    # command exits.
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm


def _connect():
    weight_shm = _attach(SHM_NAME)
    reset_shm = _attach(RESET_SHM_NAME)
    weights = np.ndarray((3,), dtype=np.float32, buffer=weight_shm.buf)
    reset_state = np.ndarray((2,), dtype=np.uint64, buffer=reset_shm.buf)
    return weight_shm, weights, reset_shm, reset_state


def _matrix_stats(candidates: list[list[float]]) -> dict:
    if not candidates:
        return {"rank": 0, "condition": None}
    matrix = np.asarray(candidates, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(matrix))
    condition = None
    if matrix.shape == (3, 3) and rank == 3:
        condition = float(np.linalg.cond(matrix))
    return {"rank": rank, "condition": condition}


def _html() -> str:
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>DIAL-MPC live weights</title>
<style>
body{font:16px system-ui;max-width:900px;margin:32px auto;padding:0 16px;background:#111827;color:#e5e7eb}
h1{font-size:24px}.row{display:grid;grid-template-columns:110px 1fr 90px;gap:14px;align-items:center;margin:18px 0}
input[type=range]{width:100%}input[type=number]{width:76px;background:#1f2937;color:white;border:1px solid #4b5563;padding:7px}
button{background:#2563eb;color:white;border:0;border-radius:6px;padding:10px 14px;margin-right:8px;cursor:pointer}
.card{background:#1f2937;border-radius:10px;padding:18px;margin-top:20px}table{width:100%;border-collapse:collapse}td,th{padding:7px;text-align:right;border-bottom:1px solid #374151}th:first-child,td:first-child{text-align:left}.good{color:#34d399}.warn{color:#fbbf24}
.viewer{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:10px}
</style></head><body>
<h1>DIAL-MPC Go2 Trot — live objective weights</h1>
<p>Changes are read by the planner on its next MPC cycle. Values are raw cost weights; they do not need to sum to one.</p>
<img id="viewer" class="viewer" alt="Start the web simulator on port 8081">
<div id="sliders"></div>
<button onclick="preset([1,1,1])">Stock gait</button>
<button onclick="preset([2,2,1])">W1</button>
<button onclick="preset([2,1,2])">W2</button>
<button onclick="preset([1,2,2])">W3</button>
<button onclick="resetSimulation()">Reset robot</button><span id="resetStatus"></span>
<button onclick="saveCandidate()">Save candidate</button>
<div class="card"><b>Candidate matrix</b> — rank <span id="rank">0</span>/3, condition <span id="condition">—</span>
<table><thead><tr><th>#</th><th>tracking</th><th>stability</th><th>gait</th></tr></thead><tbody id="candidates"></tbody></table></div>
<script>
const names=['tracking','stability','gait']; let timer;
function build(){const root=document.getElementById('sliders');names.forEach((n,i)=>{root.insertAdjacentHTML('beforeend',`<div class="row"><label>${n}</label><input id="r${i}" type="range" min="0" max="5" step="0.01" oninput="changed(${i},this.value)"><input id="n${i}" type="number" min="0" step="0.01" oninput="changed(${i},this.value)"></div>`)})}
function weights(){return names.map((_,i)=>Number(document.getElementById('n'+i).value))}
function changed(i,v){document.getElementById('r'+i).value=v;document.getElementById('n'+i).value=v;clearTimeout(timer);timer=setTimeout(update,80)}
async function update(){await fetch('/api/weights',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({weights:weights()})})}
function preset(v){v.forEach((x,i)=>changed(i,x))}
async function resetSimulation(){const s=document.getElementById('resetStatus');s.textContent=' resetting…';const r=await(await fetch('/api/reset',{method:'POST'})).json();s.textContent=r.completed?' reset complete':' reset timed out';setTimeout(()=>s.textContent='',2000)}
async function saveCandidate(){await fetch('/api/candidates',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({weights:weights()})});await refresh()}
async function refresh(){const s=await(await fetch('/api/state')).json();s.weights.forEach((x,i)=>{document.getElementById('r'+i).value=x;document.getElementById('n'+i).value=x.toFixed(3)});document.getElementById('rank').textContent=s.rank;document.getElementById('rank').className=s.rank===3?'good':'warn';document.getElementById('condition').textContent=s.condition===null?'—':s.condition.toFixed(2);document.getElementById('candidates').innerHTML=s.candidates.map((r,i)=>`<tr><td>${i}</td>${r.map(x=>`<td>${x.toFixed(3)}</td>`).join('')}</tr>`).join('')}
document.getElementById('viewer').src=`http://${location.hostname}:8081/stream.mjpg`;
build();refresh();setInterval(refresh,1000);
</script></body></html>"""


def serve(
    weights: np.ndarray,
    reset_state: np.ndarray,
    output: Path,
    host: str,
    port: int,
) -> None:
    app = Flask(__name__)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        loaded = json.loads(output.read_text(encoding="utf-8"))
        candidates = [row for row in loaded if len(row) == len(OBJECTIVE_NAMES)][-3:]
    else:
        candidates = []

    @app.get("/")
    def index():
        return _html()

    @app.get("/api/state")
    def state():
        return jsonify(
            weights=weights.copy().tolist(),
            candidates=candidates,
            **_matrix_stats(candidates),
        )

    @app.post("/api/weights")
    def set_weights():
        value = np.asarray(request.json.get("weights", []), dtype=np.float32)
        if value.shape != (3,) or not np.all(np.isfinite(value)) or np.any(value < 0):
            return jsonify(error="expected three finite non-negative weights"), 400
        weights[:] = value
        return jsonify(weights=weights.copy().tolist())

    @app.post("/api/reset")
    def reset_simulation():
        request_id = int(reset_state[0]) + 1
        reset_state[0] = np.uint64(request_id)
        deadline = time.monotonic() + 2.0
        while int(reset_state[1]) != request_id and time.monotonic() < deadline:
            time.sleep(0.005)
        return jsonify(
            reset_counter=request_id,
            completed=int(reset_state[1]) == request_id,
        )

    @app.post("/api/candidates")
    def save_candidate():
        value = np.asarray(request.json.get("weights", weights), dtype=np.float32)
        if value.shape != (3,) or not np.all(np.isfinite(value)) or np.any(value < 0):
            return jsonify(error="invalid candidate"), 400
        candidates.append(value.tolist())
        del candidates[:-3]
        output.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
        return jsonify(candidates=candidates, **_matrix_stats(candidates))

    app.run(host=host, port=port, threaded=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--serve", action="store_true", help="run browser sliders")
    action.add_argument("--set", dest="new_weights", type=_parse_weights)
    action.add_argument("--get", action="store_true", help="print current weights")
    action.add_argument(
        "--reset", action="store_true", help="reset simulator and planner"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("csm_runs/go2_weight_candidates.json"),
    )
    args = parser.parse_args()
    shm, weights, reset_shm, reset_state = _connect()
    try:
        if args.new_weights is not None:
            weights[:] = args.new_weights
            print(dict(zip(OBJECTIVE_NAMES, weights.tolist())))
        elif args.get:
            print(dict(zip(OBJECTIVE_NAMES, weights.tolist())))
        elif args.reset:
            request_id = int(reset_state[0]) + 1
            reset_state[0] = np.uint64(request_id)
            deadline = time.monotonic() + 2.0
            while int(reset_state[1]) != request_id and time.monotonic() < deadline:
                time.sleep(0.005)
            if int(reset_state[1]) != request_id:
                raise SystemExit("reset timed out; check that dial-mpc-sim is running")
            print("reset complete")
        else:
            serve(weights, reset_state, args.output, args.host, args.port)
    finally:
        shm.close()
        reset_shm.close()


if __name__ == "__main__":
    main()
