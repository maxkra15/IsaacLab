# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render recorded G1 learning baselines as one interactive, offline HTML file.

Example::

    uv run --no-sync python scripts/benchmarks/render_g1_learning_baselines.py \
        --input logs/run-a/comparison.json logs/run-b/comparison.json \
        --output logs/g1-learning-baselines/report.html

Input manifests are read without alteration. Exact completed checkpoints are
compared; missing measurements remain unavailable. Local smoke experiments and
cloud measurements are separate cohorts. GPU isolation is reported only when
declared explicitly in the experiment metadata. Adjacent command receipts expose
the preceding cold smoke separately from the fresh production process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _clean(value: Any) -> Any:
    """Make recorded nonfinite numbers unavailable in the embedded JavaScript."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _cohort(metadata: dict, source: Path) -> str:
    """Keep compute contexts explicit without inferring isolation from GPU type."""
    context = str(metadata.get("compute_context", metadata.get("benchmark_context", "")))
    if metadata.get("gpu_isolation") == "isolated" or context == "isolated_l40":
        return "Isolated L40 allocation" if "L40" in str(metadata.get("hardware", "")) else "Isolated allocation"
    if "local-" in str(source) or context in ("local_shared_gpu", "local_smoke"):
        return "Local shared-GPU smoke"
    hardware = str(metadata.get("hardware", ""))
    if "L40" in hardware:
        return "L40 run · isolation not recorded"
    return context or "Compute isolation not recorded"


def load_measurements(paths: list[Path]) -> dict:
    """Combine immutable raw inputs and retain per-run metadata and source hashes."""
    sources, runs, contexts = [], [], {}
    for path in paths:
        raw = path.read_bytes()
        document = json.loads(raw)
        if not isinstance(document.get("runs"), list):
            raise ValueError(f"{path} must contain a runs list.")
        metadata = document.get("metadata", {})
        source = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "document": document}
        sources.append(source)
        # A retrieved manifest is artifacts/runs/<run>/comparison.json. Keep
        # receipts and allocation evidence separate from the immutable manifest.
        context_dir = path.resolve().parents[2] / "metadata"
        context_files = (
            "command-receipts.jsonl",
            "runtime.json",
            "commands.json",
            "gpu-processes-before-training.json",
        )
        for name in context_files:
            context_path = context_dir / name
            if not context_path.is_file() or str(context_path) in contexts:
                continue
            context_raw = context_path.read_bytes()
            context_document = (
                [json.loads(line) for line in context_raw.splitlines() if line.strip()]
                if name.endswith(".jsonl")
                else json.loads(context_raw)
            )
            contexts[str(context_path)] = {
                "path": str(context_path),
                "sha256": hashlib.sha256(context_raw).hexdigest(),
                "document": context_document,
            }
        receipt_source = contexts.get(str(context_dir / "command-receipts.jsonl"))
        timing = _command_timing(path.parent.name, receipt_source)
        for original in document["runs"]:
            run = dict(original)
            run["report_metadata"] = metadata
            run["report_source"] = str(path.resolve())
            run["report_cohort"] = _cohort(metadata, path)
            run["report_timing"] = dict(timing)
            smoke_start, production_start = timing.get("smoke_started_unix"), run.get("process_started_unix")
            if isinstance(smoke_start, (float, int)) and isinstance(production_start, (float, int)):
                prefix = production_start - smoke_start
                if prefix >= 0:
                    run["report_timing"]["smoke_to_production_start_seconds"] = prefix
                    process_wall = run.get("process_wall_seconds")
                    if isinstance(process_wall, (float, int)):
                        run["report_timing"]["smoke_inclusive_production_seconds"] = prefix + process_wall
                    run["evaluations"] = [
                        {
                            **point,
                            "report_smoke_inclusive_wall_seconds": prefix + point["wall_time_seconds"],
                        }
                        if isinstance(point.get("wall_time_seconds"), (float, int))
                        else dict(point)
                        for point in run.get("evaluations", [])
                    ]
            # Older harness manifests may omit a health curve. Local native
            # logs can supply it; absent remote paths remain unavailable.
            if not run.get("health_curve") and run.get("log_dir"):
                log_dir = Path(run["log_dir"])
                health_file, metric_file = log_dir / "health_metrics.jsonl", log_dir / "metrics.jsonl"
                if health_file.is_file():
                    run["health_curve"] = [json.loads(line) for line in health_file.read_text().splitlines() if line]
                elif metric_file.is_file():
                    rows = [json.loads(line) for line in metric_file.read_text().splitlines() if line]
                    run["health_curve"] = [
                        {"iteration": row["iteration"], **row["health"]} for row in rows if "health" in row
                    ]
            runs.append(run)
    return _clean({"sources": sources, "contexts": list(contexts.values()), "runs": runs})


def _command_timing(run_directory: str, receipt_source: dict | None) -> dict:
    """Derive command durations only for the production run named in a receipt."""
    if receipt_source is None:
        return {}
    receipts = receipt_source["document"]

    def targets_run(receipt: dict) -> bool:
        argv = receipt.get("argv", [])
        return "--output" in argv and Path(argv[argv.index("--output") + 1]).name == run_directory

    trains = [row for row in receipts if row.get("name") == "train" and targets_run(row)]
    if not trains:
        return {}
    train = trains[-1]
    timing = {"receipt_path": receipt_source["path"], "receipt_sha256": receipt_source["sha256"]}
    smokes = [
        row
        for row in receipts
        if row.get("name") == "smoke" and row.get("finished_unix", math.inf) <= train.get("started_unix", -math.inf)
    ]
    evaluations = [row for row in receipts if row.get("name") == "evaluate" and targets_run(row)]
    selected = {"train_command": train}
    if smokes:
        selected["smoke"] = smokes[-1]
    if evaluations:
        selected["evaluate_command"] = evaluations[-1]
    for name, receipt in selected.items():
        start, finish = receipt.get("started_unix"), receipt.get("finished_unix")
        if isinstance(start, (float, int)) and isinstance(finish, (float, int)) and finish >= start:
            timing[name + "_seconds"] = finish - start
            timing[name + "_started_unix"] = start
            timing[name + "_finished_unix"] = finish
            timing[name + "_exit_code"] = receipt.get("exit_code")
    if "smoke_started_unix" in timing and "train_command_finished_unix" in timing:
        timing["smoke_and_train_command_seconds"] = timing["train_command_finished_unix"] - timing["smoke_started_unix"]
    return timing


def render_report(measurements: dict) -> str:
    """Create an offline report with quality, health, timing, and audit views."""
    payload = json.dumps(measurements, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return _TEMPLATE.replace("__DATA__", payload)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 · learning baselines</title><style>
:root{color-scheme:light;--ink:#183047;--muted:#61748a;--line:#dce5ed;--accent:#416cba}
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1450px;margin:auto;padding:36px 26px}h1{font-size:38px;letter-spacing:-1px;margin:5px 0 12px}
h2{font-size:20px;margin:0 0 14px}h3{font-size:15px;margin:18px 0 8px}p{margin:8px 0 15px}.muted{color:var(--muted)}
.small{font-size:12px}.eyebrow{font-size:12px;font-weight:650;letter-spacing:2px;color:var(--accent)}
.panel,.card{background:white;border:1px solid var(--line);border-radius:14px;padding:22px;margin-top:20px}
.controls{display:flex;gap:15px;flex-wrap:wrap;align-items:center}.controls label{font-size:13px;color:var(--muted)}
select,button{font:inherit;color:var(--ink);border:1px solid var(--line);background:white;padding:7px
9px;border-radius:7px}
select{margin-left:5px}button,summary{cursor:pointer}input{accent-color:var(--accent)}.legend label{margin-right:15px}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{margin-top:16px;padding:17px}
.value{font-size:25px;font-weight:700;margin:8px 0}.badge{display:inline-block;background:#e5eef5;border-radius:16px;
padding:4px 9px;font-size:12px;margin:4px 6px 0 0}.notice{border-left:4px solid #be853e;padding:13px 17px;
background:#fff7e9;border-radius:8px;margin-top:16px}.grid{display:grid;grid-template-columns:1.3fr 1fr;gap:20px}
.grid>.panel{min-width:0}.chart{width:100%;height:auto;display:block}.chart text{font-size:12px;fill:var(--muted)}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;
font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:11px 10px;border-bottom:1px solid
var(--line)}th{font-size:11px;text-transform:uppercase;color:var(--muted)}
td:first-child{font-weight:600}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
details{border-top:1px solid var(--line);padding:13px 0}summary{font-weight:600}pre{font:12px/1.6
ui-monospace,monospace;
background:#f3f6f9;border-radius:8px;padding:14px;white-space:pre-wrap;overflow-wrap:anywhere;
max-height:600px;overflow:auto}
.tooltip{display:none;position:fixed;background:var(--ink);color:white;white-space:pre-line;font-size:12px;padding:12px;
border-radius:8px;max-width:350px;z-index:10;pointer-events:none}a{color:var(--accent)}
footer{margin-top:25px;color:var(--muted);font-size:12px}
@media(max-width:1050px){.grid{grid-template-columns:1fr}}@media(max-width:650px){main{padding:22px
13px}h1{font-size:30px}
.panel{padding:16px}.cards{grid-template-columns:1fr}.controls{gap:10px}}
</style></head><body><main>
<header><div class="eyebrow">ISAAC LAB · RSL-RL · ROBOLEARN</div><h1>G1 learning baselines</h1>
<p class="muted">Stock RSL-RL PPO, matching Warp PPO, and the FlashSAC authors' recipe.</p>
<span class="badge">Recorded measurements</span><span class="badge">Offline and self-contained</span>
<span class="badge" id="inventory"></span>
<p class="notice" id="cohort-note"></p></header>
<section class="panel"><div class="controls">
<div class="legend" id="algorithms"></div>
<label>Compute cohort<select id="cohort"></select></label>
<label>Seed<select id="seed"><option value="all">All seeds</option></select></label>
<label>Scenario<select id="scenario"><option value="forward_0_5">Forward 0.5 m/s</option>
<option value="native_commands">Native velocity commands</option></select></label></div>
<div class="controls" style="margin-top:14px">
<label>Checkpoint<select id="checkpoint"></select></label>
<label>Horizontal axis<select id="x"><option value="wall_time_seconds">Production checkpoint age (s)</option>
<option value="report_smoke_inclusive_wall_seconds">Smoke-inclusive checkpoint age (s)</option>
<option value="transitions">Collected transitions</option><option value="time_seconds">Training loop wall (s)</option>
<option value="iteration">Completed iterations</option></select></label>
<label>Quality<select id="metric"><option value="walking_success_rate">Walking success (%)</option>
<option value="return_mean">Episode return</option><option value="linear_velocity_error_mean">Planar error
(m/s)</option>
<option value="yaw_velocity_error_mean">Yaw error (rad/s)</option><option value="survival_rate">Survival (%)</option>
<option value="forward_velocity_mean">Forward velocity (m/s)</option><option value="upright_fraction">Upright
(%)</option>
</select></label><button id="download">Download embedded measurements</button></div>
<p class="small muted">Tables compare exactly the selected completed iteration. Missing checkpoints remain unavailable;
curves connect observations without estimating intermediate values. Hover chart dots for raw measurements.</p></section>
<div class="cards" id="cards"></div>
<div class="grid"><section class="panel"><h2 id="quality-title">Common native evaluation</h2>
<svg id="quality" class="chart" viewBox="0 0 850 400" role="img" aria-label="Common native policy quality"></svg>
<p class="small muted">Deterministic full native episodes. Velocity metrics describe the scored portion before timeout
or failure; inspect survival alongside error. Training rewards use different logging windows and are not these
evaluations.</p>
</section><section class="panel"><h2>Production process wall and phases</h2>
<svg id="phases" class="chart" viewBox="0 0 620 400" role="img" aria-label="Process wall and timing phases"></svg>
<p class="small muted">Outline: production training subprocess wall. Blue: rollout; teal: update; amber: explicit Warp
preparation; gray: other production process time. This wall includes production startup, residual compilation, logging,
checkpoints, and shutdown. It excludes the preceding cold smoke, which can compile and prewarm caches; see the command
timings below. Evaluation is separate. Flash interleaved phases use CUDA
events,
while PPO phases use synchronized wall timers. Phase ratios do not isolate a kernel optimization.</p></section></div>
<section class="panel"><h2>Exact checkpoint evaluations</h2><div class="scroll"><table><thead><tr>
<th>Learner / seed</th><th>Status</th><th>Iteration</th><th>Return</th><th>XY error m/s</th><th>Yaw error rad/s</th>
<th>Survival</th><th>Walking</th><th>Forward m/s</th><th>Gate</th><th>Production age s</th><th>Transitions</th>
</tr></thead><tbody id="evaluations"></tbody></table></div>
<h3>Seed means and observed ranges</h3><div class="scroll"><table><thead><tr><th>Learner</th><th>Distinct seeds</th>
<th>Return mean [min, max]</th><th>XY error mean [min, max]</th><th>Survival mean [min, max]</th>
<th>Walking mean [min, max]</th><th>Passing gates</th></tr></thead><tbody id="seed-summary"></tbody></table></div>
<p class="small muted">Ranges describe observed training-seed spread, not confidence intervals. Each cell uses only
available exact checkpoint measurements. A small seed count cannot establish a stable algorithm ranking.</p></section>
<section class="panel"><h2>Completed production speed · seed means and ranges</h2>
<div class="scroll"><table><thead><tr><th>Learner</th><th>Completed planned seeds</th><th>Selected completed seeds</th>
<th>Transitions per seed</th><th>Loop transitions/s</th><th>Production transitions/s</th>
<th>Production wall s</th><th>Smoke-inclusive wall s</th></tr></thead><tbody id="speed-summary"></tbody></table></div>
<p class="small muted">The cloud protocol plans seeds 0, 1, and 2. Coverage counts distinct completed runs in the
selected cohort before the seed filter; speed means and [min, max] use only the selected completed seeds.
Local smoke runs have no planned three-seed production coverage. Loop speed uses recorded iteration timers;
production speed includes production subprocess overhead. Logger and checkpoint I/O can contribute to differences.
Missing or unfinished runs are excluded; they do not count as zero-speed measurements.</p></section>
<section class="panel"><h2>First observed forward walking gate · per seed</h2>
<div class="scroll"><table><thead><tr><th>Learner / seed</th><th>Observation</th><th>Iteration</th><th>Transitions</th>
<th>Production age s</th><th>Smoke-inclusive age s</th><th>Previous failed iteration</th>
<th>Failed → pass transition bracket</th><th>Failed → pass production age bracket s</th>
<th>Later failed checkpoints</th></tr></thead><tbody id="first-gates"></tbody></table></div>
<p class="small muted" id="final-gate-note"></p>
<p class="small muted">This view uses all recorded forward checkpoints, independently of the scenario/checkpoint
selector. A first observed pass identifies that saved policy's evaluation, not an exact threshold crossing.
Brackets join the previous recorded failure and first recorded pass; no interpolation or monotonic improvement is
assumed. No observed pass is right-censored at the last checkpoint with valid gate metrics. A pass can later regress.
Checkpoint age excludes the common evaluation process, which runs after training; smoke-inclusive age also includes
the preceding smoke and harness launch delay, starting at the receipt's smoke start.</p></section>
<section class="panel"><h2>Cold smoke, fresh production, and evaluation command costs</h2>
<div class="scroll"><table><thead><tr><th>Learner / seed</th><th>Preceding smoke s</th><th>Production subprocess s</th>
<th>Train wrapper s</th><th>Smoke → production end s</th><th>Smoke → train wrapper end s</th>
<th>Evaluation wrapper s</th><th>Receipt SHA256</th></tr></thead><tbody id="cold-costs"></tbody></table></div>
<p class="small muted">Each cloud workflow first executes a short smoke, then launches a fresh production learner.
The smoke includes cold preparation/compilation and short validation training; it can prewarm disk caches for
production. Receipt durations measure whole commands, not isolated compile kernels. Smoke transitions are excluded
from the production budget and throughput denominator. Smoke-inclusive wall adds its real elapsed overhead;
evaluation remains separate. These timings start at smoke, excluding earlier workflow provisioning and setup.
Absent receipts stay unavailable, and their hashes are separate from the original comparison JSON hashes.</p></section>
<section class="panel"><div class="controls"><h2 style="margin:0">Exploration and optimizer health</h2>
<label>Metric<select id="health-metric"><option value="std_mean">Mean Gaussian action std</option>
<option value="std_min">Minimum Gaussian action std</option><option value="std_max">Maximum Gaussian action std</option>
<option value="learning_rate">PPO learning rate</option><option value="kl">PPO last minibatch KL</option>
<option value="actor_grad_norm">Warp actor gradient norm</option><option value="critic_grad_norm">Warp critic
gradient norm</option>
<option value="actor/entropy">FlashSAC actor entropy</option><option value="temperature/value">FlashSAC
temperature</option>
</select></label></div><svg id="health" class="chart" viewBox="0 0 1200 330" role="img" aria-label="Recorded learner
health"></svg>
<p class="small muted">Horizontal axis: completed iterations. PPO telemetry refers to its Gaussian action distribution;
FlashSAC uses a tanh policy and adaptive entropy, so Gaussian std is not a shared measurement.
Warp losses/KL/gradient norms describe the last minibatch. Missing telemetry is unavailable.</p></section>
<section class="panel"><h2>Actual budgets and weight evidence</h2><div class="scroll"><table><thead><tr>
<th>Learner / seed</th><th>Transitions</th><th>Actor calls</th><th>Critic calls</th><th>Actual actor steps</th>
<th>Actual critic steps</th><th>Finite state</th><th>Updated weights</th><th>Production wall s</th><th>Production
transitions/s</th>
</tr></thead><tbody id="budgets"></tbody></table></div>
<p class="small muted">This table uses each run's final recorded training budget. Flash AMP can skip an optimizer step:
attempted update calls and checkpoint Adam steps are reported separately. PPO samples each rollout across epochs;
FlashSAC reuses replay data. Matching collection budgets does not match optimizer work.</p></section>
<section class="panel"><h2>Native foot-contact diagnostics</h2><div class="scroll"><table><thead><tr>
<th>Learner / seed</th><th>Flight</th><th>Single support</th><th>Double support</th><th>Alternating touchdowns</th>
<th>Foot touchdown means</th></tr></thead><tbody id="contacts"></tbody></table></div>
<p class="small muted">Recorded native contact forces use the logged threshold. Alternation alone does not prove
walking; forward tracking and survival are the primary gate. Per-world contact traces, when recorded, remain in the
raw details. The 1 N threshold can count rapid small-force crossings as touchdowns; high alternation does not establish
a natural gait.</p></section>
<section class="panel"><h2>Execution path and interpretation</h2>
<p>The primary comparison uses the existing Torch MDP: native Isaac Lab observations, rewards, actions, resets,
and commands remain eager. Newton/MuJoCo Warp physics runs in its own graph. Warp PPO's learner update is a separate
captured graph; RSL-RL PPO uses Torch; FlashSAC uses compiled Torch networks and AMP when recorded as enabled.
This is not an end-to-end differentiable physics experiment. Any additional captured-MDP variants must be identified
in metadata.</p>
<p>Both PPO policies retain native unclipped Gaussian actions. FlashSAC uses the authors' bounded policy support and
the same environment joint-target scaling. Policy support, initialization RNG, and numerical precision can differ.
Lower velocity error must be considered together with survival, forward velocity, and the exact evaluation scenario.</p>
<p>These are equal production collection budgets of 50.38 million transitions per full run. The paper's GPU budget is
50 million for FlashSAC and 200 million for PPO; it reports estimated wall time from profiling and uses Isaac Lab 2.1.0
with PhysX, whereas these runs use Isaac Lab 3 with Newton/MuJoCo Warp.
This comparison follows the authors' launch recipe rather than claiming an exact paper reproduction: their
launch script uses n-step 3 and two replay updates per 1,024-environment vector step; Table 9 uses n-step 1 and two
updates per 2,048-environment step.
<a href="https://arxiv.org/html/2604.04539v2#S8">Paper Appendix 8</a> ·
<a href="https://github.com/Holiday-Robot/FlashSAC/blob/main/scripts/run_isaaclab.sh">Authors' launch script</a>.</p>
<p id="initialization-note"></p>
<p>Three seeds provide descriptive means and ranges, not a causal learning-quality claim; policy initialization and
sampling RNG also differ. The selected compute cohort and immutable input paths identify the source of every row.</p>
<p>The forward walking gate requires walking success ≥80%, mean planar error ≤0.2 m/s, and survival ≥90%.
Each episode's walking criterion requires survival, planar error &lt;0.25 m/s, yaw error &lt;0.4 rad/s, and forward
velocity
&gt;0.25 m/s. Gate values come from the recorded protocol; absent metrics never pass.</p>
<div class="scroll"><table><thead><tr><th>Learner / seed</th><th>Hardware / cohort</th><th>Policy support</th>
<th>Recipe</th><th>Episode counter init</th><th>Compile / AMP</th><th>MDP hash</th><th>Source revisions</th>
</tr></thead><tbody
id="recipes"></tbody></table></div>
<h3>Recorded sources, commands, configurations, and raw evidence</h3><div id="details"></div></section>
<footer>PPO: <a href="https://arxiv.org/abs/1707.06347">Schulman, Wolski, Dhariwal, Radford, and Klimov (2017)</a>.
FlashSAC: <a href="https://arxiv.org/abs/2604.04539">Donghu Kim, Youngdo Lee, and coauthors (2026)</a>,
<a href="https://github.com/Holiday-Robot/FlashSAC">official implementation</a>.
Learners: <a href="https://github.com/leggedrobotics/rsl_rl">RSL-RL</a>,
<a href="https://github.com/maxkra15/RoboLearn">RoboLearn</a>, and
<a href="https://nvidia.github.io/warp-nn/">NVIDIA Warp-NN</a>.
No algorithm is assumed faster or better before measuring learning quality and elapsed time.</footer>
</main><div class="tooltip" id="tooltip"></div><script type="application/json" id="data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),runs=data.runs||[],$=id=>document.getElementById(id);
const finite=v=>typeof v==='number'&&Number.isFinite(v),fmt=(v,d=2)=>finite(v)?v.toLocaleString(undefined,
{minimumFractionDigits:d,maximumFractionDigits:d}):'—',pct=v=>finite(v)?fmt(v*100,1)+'%':'—';
const algorithms=[...new Set(runs.map(r=>r.algorithm||'unknown'))],enabled=new Set(algorithms);
const names={rsl_rl_ppo:'RSL-RL PPO',warp_ppo:'Warp PPO',flashsac:'FlashSAC'},palette=['#416cba','#087e74',
'#a57536','#9861ad'];
const color=r=>palette[algorithms.indexOf(r.algorithm)%palette.length],
label=r=>(names[r.algorithm]||r.name||r.algorithm)+' · seed '+r.seed;
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const esc=s=>String(s).replace(/[&<>"']/g,c=>
({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const spec={walking_success_rate:['Walking success (%)',100],return_mean:['Episode return',1],
linear_velocity_error_mean:['Planar velocity error (m/s)',1],yaw_velocity_error_mean:['Yaw error (rad/s)',1],
survival_rate:['Survival (%)',100],forward_velocity_mean:['Forward velocity (m/s)',1],
upright_fraction:['Upright (%)',100]};
const axes={wall_time_seconds:'Production checkpoint age (s)',time_seconds:'Training loop wall (s)',
report_smoke_inclusive_wall_seconds:'Smoke-inclusive checkpoint age (s)',
transitions:'Collected transitions',iteration:'Completed iterations'};
const scenarios=r=>(r.evaluations||[]).filter(p=>p.scenario===$('scenario').value);
function visible(){return runs.filter(r=>enabled.has(r.algorithm)&&r.report_cohort===$('cohort').value&&
($('seed').value==='all'||String(r.seed)===$('seed').value));}
function selected(r){return scenarios(r).find(p=>p.iteration===Number($('checkpoint').value));}
function gate(p,r){if(!p||p.scenario!=='forward_0_5')return null;const g=r.report_metadata.walking_gate||{};
if(!finite(p.walking_success_rate)||!finite(p.linear_velocity_error_mean)||!finite(p.survival_rate))return null;
return p.walking_success_rate>=(g.walking_success_rate_min??.8)&&
p.linear_velocity_error_mean<=(g.linear_velocity_error_mean_max??.2)&&
p.survival_rate>=(g.survival_rate_min??.9);}
function option(parent,value,text){const o=document.createElement('option');o.value=value;o.textContent=text;
parent.append(o);}
function row(parent,items){const tr=document.createElement('tr');
items.forEach(item=>{const td=document.createElement('td');td.textContent=item;tr.append(td);});parent.append(tr);}
function badgeName(r){return '<span class="dot" style="background:'+color(r)+'"></span>'+esc(label(r));}
function tooltip(event,text){const box=$('tooltip');box.textContent=text;box.style.display='block';
box.style.left=Math.max(10,Math.min(event.clientX+12,innerWidth-box.offsetWidth-10))+'px';
box.style.top=Math.max(10,Math.min(event.clientY+12,innerHeight-box.offsetHeight-10))+'px';}
const ns='http://www.w3.org/2000/svg';function svg(parent,tag,attrs,text){const n=document.createElementNS(ns,tag);
Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;parent.append(n);
return n;}
function lineChart(id,series,xlabel,ylabel,width,height){const s=$(id);s.replaceChildren();
const points=series.flatMap(t=>t.points);
if(!points.length){svg(s,'text',{x:width/2,y:height/2,'text-anchor':'middle'},
'No recorded values for this selection');return;}
const l=70,r=25,t=24,b=55,xmin=0,xmax=Math.max(...points.map(p=>p.x),1),values=points.map(p=>p.y);
let ymin=Math.min(...values),ymax=Math.max(...values);const pad=(ymax-ymin||Math.max(Math.abs(ymax),1))*.12;
ymin-=pad;ymax+=pad;
const X=v=>l+(v-xmin)/(xmax-xmin)*(width-l-r),Y=v=>t+(ymax-v)/(ymax-ymin)*(height-t-b);
for(let i=0;i<=4;i++){const xv=xmax*i/4,yv=ymin+(ymax-ymin)*i/4;
svg(s,'line',{x1:l,y1:Y(yv),x2:width-r,y2:Y(yv),stroke:'#e4eaf0'});
svg(s,'text',{x:l-10,y:Y(yv)+4,'text-anchor':'end'},fmt(yv,Math.abs(yv)<10?2:0));
svg(s,'text',{x:X(xv),y:height-b+22,'text-anchor':'middle'},xv>=1e6?fmt(xv/1e6,1)+'M':fmt(xv,0));}
svg(s,'text',{x:width/2,y:height-8,'text-anchor':'middle'},xlabel);svg(s,'text',{x:l,y:15},ylabel);
series.forEach(a=>{const pts=a.points.slice().sort((p,q)=>p.x-q.x);svg(s,'polyline',
{points:pts.map(p=>X(p.x)+','+Y(p.y)).join(' '),fill:'none',stroke:color(a.run),'stroke-width':2,opacity:.8});
pts.forEach(p=>{const n=svg(s,'circle',{cx:X(p.x),cy:Y(p.y),r:4,fill:color(a.run),stroke:'white','stroke-width':1});
n.addEventListener('mousemove',e=>tooltip(e,label(a.run)+'\n'+JSON.stringify(p.raw,null,2)));
n.addEventListener('mouseleave',()=>{$('tooltip').style.display='none';});});});}
function checkpointOptions(){const old=$('checkpoint').value;const values=[...new Set(visible().flatMap(r=>
scenarios(r).map(p=>p.iteration)).filter(finite))].sort((a,b)=>a-b);$('checkpoint').replaceChildren();
values.forEach(v=>option($('checkpoint'),v,v.toLocaleString()+' completed'));
if(!values.length){option($('checkpoint'),'','No evaluations recorded');return;}
const sets=visible().filter(r=>scenarios(r).length).map(r=>new Set(scenarios(r).map(p=>p.iteration)));
const common=values.filter(v=>sets.every(s=>s.has(v)));
$('checkpoint').value=values.includes(Number(old))?old:String((common.length?common:values).at(-1));}
function quality(){const key=$('metric').value,[title,scale]=spec[key],x=$('x').value;
$('quality-title').textContent=title;
lineChart('quality',visible().map(run=>({run,points:scenarios(run).filter(p=>finite(p[x])&&finite(p[key])).map(p=>
({x:p[x],y:p[key]*scale,raw:p}))})),axes[x],title,850,400);}
function health(){const key=$('health-metric').value,aliases={std_mean:'action_std_mean',std_min:'action_std_min',
std_max:'action_std_max'};
lineChart('health',visible().map(run=>({run,points:(run.health_curve||[]).map(p=>({x:p.iteration,
y:p[key]??p[aliases[key]],raw:p}))
.filter(p=>finite(p.x)&&finite(p.y))})),'Completed iterations',$('health-metric').selectedOptions[0].textContent,
1200,330);}
function evaluations(){const table=$('evaluations');table.replaceChildren();
visible().forEach(r=>{const p=selected(r)||{},g=gate(selected(r),r);
row(table,[label(r),r.status||'unknown',fmt(p.iteration,0),fmt(p.return_mean),fmt(p.linear_velocity_error_mean,3),
fmt(p.yaw_velocity_error_mean,3),pct(p.survival_rate),pct(p.walking_success_rate),fmt(p.forward_velocity_mean,3),
g===null?'—':g?'PASS':'FAIL',fmt(p.wall_time_seconds,1),fmt(p.transitions,0)]);});
const summary=$('seed-summary'),cards=$('cards');summary.replaceChildren();cards.replaceChildren();
algorithms.forEach(a=>{
const records=visible().filter(r=>r.algorithm===a&&selected(r)),unique=new Map(records.map(r=>[r.seed,r]));
const rr=[...unique.values()],pp=rr.map(selected);if(!records.length)return;
const range=(key,percentage=false)=>{const v=pp.map(p=>p[key]).filter(finite);if(!v.length)return '—';
const f=percentage?pct:n=>fmt(n,3);return f(mean(v))+' ['+f(Math.min(...v))+', '+f(Math.max(...v))+']';};
const passes=rr.filter(r=>gate(selected(r),r)===true).length;
row(summary,[names[a]||a,rr.length,range('return_mean'),range('linear_velocity_error_mean'),range('survival_rate',true),
range('walking_success_rate',true),$('scenario').value==='forward_0_5'?passes+'/'+rr.length:'—']);
const c=document.createElement('div');c.className='card';const walking=pp.map(p=>p.walking_success_rate).filter(finite);
c.innerHTML='<div>'+esc(names[a]||a)+'</div><div class="value">'+(walking.length?pct(mean(walking)):'—')+'</div>'+
'<div class="muted small">Mean walking success · '+rr.length+' distinct seed(s)<br>XY error '+
esc(range('linear_velocity_error_mean'))+
' m/s<br>Survival '+esc(range('survival_rate',
true))+'<br>Walking gates '+($('scenario').value==='forward_0_5'?passes+'/'+rr.length:'not applicable')+'</div>';
if(records.length!==unique.size)c.innerHTML+='<p class="small">Repeated seed runs exist; '+
'seed summaries use the last recorded run per seed.</p>';cards.append(c);});}
function distinctSeeds(rr){return [...new Map(rr.map(r=>[r.algorithm+':'+r.seed,r])).values()];}
function meanRange(values,d=2){const vv=values.filter(finite);if(!vv.length)return '—';
return fmt(mean(vv),d)+' ['+fmt(Math.min(...vv),d)+', '+fmt(Math.max(...vv),d)+']';}
function speedSummary(){const t=$('speed-summary');t.replaceChildren();
const rate=(r,k)=>finite(r.transitions)&&finite(r[k])&&r[k]>0?r.transitions/r[k]:null;
algorithms.filter(a=>enabled.has(a)).forEach(a=>{
const all=distinctSeeds(runs.filter(r=>r.algorithm===a&&r.report_cohort===$('cohort').value)),
done=all.filter(r=>r.status==='completed'),rr=distinctSeeds(visible().filter(r=>r.algorithm===a))
.filter(r=>r.status==='completed');if(!all.length)return;
const planned=[0,1,2],completed=done.filter(r=>planned.includes(r.seed)),
missing=planned.filter(seed=>!completed.some(r=>r.seed===seed));
const coverage=$('cohort').value.includes('Local')?'Smoke · no production seed plan':
completed.length+'/3'+(missing.length?' · missing '+missing.join(', '):' · all recorded');
const cold=rr.map(r=>r.report_timing?.smoke_inclusive_production_seconds),available=cold.filter(finite).length;
row(t,[names[a]||a,coverage,rr.map(r=>r.seed).join(', ')||'None',meanRange(rr.map(r=>r.transitions),0),
meanRange(rr.map(r=>rate(r,'training_seconds')),0),meanRange(rr.map(r=>rate(r,'process_wall_seconds')),0),
meanRange(rr.map(r=>r.process_wall_seconds),1),meanRange(cold,1)+
(available<rr.length?' · '+available+'/'+rr.length+' timings':'')]);});}
function firstGates(){const t=$('first-gates');t.replaceChildren();const rr=distinctSeeds(visible());
rr.forEach(r=>{
const points=(r.evaluations||[]).filter(p=>p.scenario==='forward_0_5'&&finite(p.iteration)&&gate(p,r)!==null)
.slice().sort((a,b)=>a.iteration-b.iteration),first=points.find(p=>gate(p,r)===true),
last=points.at(-1),observed=first||last,previous=first?points.filter(p=>p.iteration<first.iteration&&
gate(p,r)===false).at(-1):null,later=first?points.filter(p=>p.iteration>first.iteration&&gate(p,r)===false):[];
const bracket=key=>previous&&finite(previous[key])&&finite(first[key])?
'('+fmt(previous[key],key==='transitions'?0:1)+', '+fmt(first[key],key==='transitions'?0:1)+']':
first?'No earlier recorded failure':'Right-censored';
row(t,[label(r),first?'First observed PASS':last?'No pass observed · right-censored':'No valid gate evaluations',
fmt(observed?.iteration,0),fmt(observed?.transitions,0),fmt(observed?.wall_time_seconds,1),
fmt(observed?.report_smoke_inclusive_wall_seconds,1),fmt(previous?.iteration,0),
observed?bracket('transitions'):'—',observed?bracket('wall_time_seconds'):'—',
first?(later.map(p=>fmt(p.iteration,0)).join(', ')||'None recorded'):'—']);});
const summaries=algorithms.filter(a=>a.includes('ppo')).map(a=>{
const completed=rr.filter(r=>r.algorithm===a&&r.status==='completed'),last=completed.map(r=>({r,p:
(r.evaluations||[]).filter(p=>p.scenario==='forward_0_5').slice().sort((p,q)=>p.iteration-q.iteration).at(-1)}));
if(!last.length)return null;return (names[a]||a)+' '+last.filter(({r,p})=>gate(p,r)===true).length+
'/'+last.length+' pass at their final recorded forward checkpoint';}).filter(Boolean);
$('final-gate-note').textContent=summaries.join('; ');}
function coldCosts(){const t=$('cold-costs');t.replaceChildren();visible().forEach(r=>{const c=r.report_timing||{};
row(t,[label(r),fmt(c.smoke_seconds,1),fmt(r.process_wall_seconds,1),fmt(c.train_command_seconds,1),
fmt(c.smoke_inclusive_production_seconds,1),fmt(c.smoke_and_train_command_seconds,1),
fmt(c.evaluate_command_seconds,1),c.receipt_sha256||'Unavailable']);});}
function phases(){const s=$('phases');s.replaceChildren();const rr=visible(),max=Math.max(...rr.map(r=>
Math.max(r.process_wall_seconds||0,(r.rollout_seconds||0)+(r.update_seconds||0)+(r.warmup_seconds||0))),1);
const height=Math.max(260,rr.length*54+90);s.setAttribute('viewBox','0 0 620 '+height);
rr.forEach((r,i)=>{const y=30+i*54;svg(s,'text',{x:10,y:y},label(r));const total=r.process_wall_seconds;
if(!finite(total)){svg(s,'text',{x:400,y:y},'No process timing');return;}
const parts=[r.rollout_seconds||0,r.update_seconds||0,r.warmup_seconds||0],sum=parts.reduce((a,b)=>a+b,0);
parts.push(Math.max(0,total-sum));let start=10;parts.forEach((v,j)=>{const w=v/max*490;svg(s,'rect',
{x:start,y:y+8,width:w,height:15,fill:['#416cba','#087e74','#c79240','#a6b2bf'][j]});start+=w;});
svg(s,'rect',{x:10,y:y+8,width:total/max*490,height:15,fill:'none',stroke:'#183047','stroke-width':1.5});
svg(s,'text',{x:510,y:y+20},fmt(total,1)+' s');if(sum>total+1)svg(s,'text',{x:10,y:y+40},
'Recorded phase sum exceeds process wall; inspect timing scope');});
svg(s,'text',{x:10,y:height-15},'Production subprocess wall; preceding cold smoke excluded');}
function budgets(){const t=$('budgets');t.replaceChildren();visible().forEach(r=>{const e=r.learning_evidence||{};
const actual=n=>finite(e[n+'_optimizer_steps_min'])?fmt(e[n+'_optimizer_steps_min'],
0)+'–'+fmt(e[n+'_optimizer_steps_max'],0):fmt(e.optimizer_steps,0);
const yes=v=>v===true?'Yes':v===false?'No':'—';row(t,[label(r),fmt(r.transitions,0),fmt(r.actor_gradient_updates,0),
fmt(r.critic_gradient_updates,0),actual('actor'),actual('critic'),yes(e.weights_and_optimizer_finite),
yes(e.weights_updated),
fmt(r.process_wall_seconds,1),finite(r.transitions)&&finite(r.process_wall_seconds)&&r.process_wall_seconds>0?
fmt(r.transitions/r.process_wall_seconds,0):'—']);});}
function contacts(){const t=$('contacts');t.replaceChildren();visible().forEach(r=>{const p=selected(r),
c=p?.foot_contacts||{};
row(t,[label(r),pct(c.flight_fraction),pct(c.single_support_fraction),pct(c.double_support_fraction),
pct(c.alternating_touchdown_fraction),
(c.touchdowns_per_episode_mean||[]).map(n=>fmt(n,2)).join(' / ')||'—']);});}
function recipes(){const t=$('recipes');t.replaceChildren();visible().forEach(r=>{const m=r.report_metadata,
c=r.agent_config||{},a=c.algorithm_cfg||c.algorithm||{};
const support=c.clip_actions===null?'Unclipped Gaussian':finite(c.clip_actions)?
'Policy clip ±'+c.clip_actions:'Not recorded';
let recipe=r.algorithm==='flashsac'?'n-step '+(a.n_step??'—')+', replay batch '+(a.sample_batch_size??'—'):
(a.activation||c.actor?.activation||'—')+', '+(a.std_type||c.actor?.distribution_cfg?.std_type||'—')+' std, '+
(a.epochs||a.num_learning_epochs||'—')+' epochs × '+(a.num_mini_batches||'—')+' batches, '+(a.schedule||'—')+' LR';
row(t,[label(r),(m.hardware||'—')+' / '+r.report_cohort,support,recipe,
c.init_at_random_ep_len===true?'Random':c.init_at_random_ep_len===false?'Zero · diagnostic':'Not recorded in config',
r.algorithm==='flashsac'?'Compile '+String(a.use_compile??'—')+' / AMP '+String(a.use_amp??'—'):'—',
(m.mdp_sha256||'—').slice(0,14),(m.isaaclab_revision||'—').slice(0,12)+' / '+(m.robolearn_revision||'—').slice(0,
12)]);});
const flash=distinctSeeds(visible().filter(r=>r.algorithm==='flashsac')),bad=flash.filter(r=>
r.agent_config?.init_at_random_ep_len===false),matched=flash.filter(r=>r.agent_config?.init_at_random_ep_len===true),
unknown=flash.length-bad.length-matched.length,notes=[];
if(bad.length)notes.push(bad.length+' selected FlashSAC run(s) use original zero episode counters. '+
'These are initialization diagnostics: PPO and the authors\' wrapper use random episode lengths. '+
'The corrected primary comparison uses init_at_random_ep_len=True.');
if(matched.length)notes.push(matched.length+' selected FlashSAC run(s) record random episode-counter '+
'initialization, matching PPO and the authors\' wrapper.');
if(unknown)notes.push(unknown+' selected FlashSAC run(s) do not record the episode-counter initialization flag.');
$('initialization-note').textContent=notes.join(' ');
$('initialization-note').className=bad.length?'notice':'muted';}
function details(){const d=$('details');d.replaceChildren();
const evidence=[...data.sources.map(s=>({...s,kind:'Raw comparison'})),
...(data.contexts||[]).map(s=>({...s,kind:'Separate receipt/context'}))];
evidence.forEach(s=>{const item=document.createElement('details');
const heading=document.createElement('summary');heading.textContent=s.kind+' · '+s.path+' · SHA256 '+s.sha256;
const pre=document.createElement('pre');
pre.textContent=JSON.stringify(s.document,null,2);item.append(heading,pre);d.append(item);});}
function refresh(){quality();health();evaluations();speedSummary();firstGates();coldCosts();phases();budgets();
contacts();recipes();
const cohort=$('cohort').value;
$('cohort-note').textContent=cohort.includes('Local')?
'Local shared-GPU smoke: short runs validate execution and checkpoint playback. '+
'They do not establish learning quality or an isolated speed ranking.':
cohort.includes('not recorded')?'GPU isolation is not recorded. '+
'Inspect hardware, allocation, budgets, and learning quality before comparing speed.':
'Recorded compute cohort: '+cohort+'. Compare exact budgets, several seeds, '+
'and common walking quality before drawing a speed or learning ranking.';}
algorithms.forEach(a=>{const l=document.createElement('label'),i=document.createElement('input');i.type='checkbox';
i.checked=true;
i.addEventListener('change',()=>{i.checked?enabled.add(a):enabled.delete(a);checkpointOptions();refresh();});
l.append(i,' '+(names[a]||a));$('algorithms').append(l);});
const cohorts=[...new Set(runs.map(r=>r.report_cohort))];cohorts.forEach(c=>option($('cohort'),c,c));
$('cohort').value=cohorts.find(c=>c.includes('Isolated'))||cohorts[0]||'';
[...new Set(runs.map(r=>r.seed))].sort((a,b)=>a-b).forEach(s=>option($('seed'),s,'Seed '+s));
['cohort','seed','scenario'].forEach(id=>$(id).addEventListener('change',()=>{checkpointOptions();refresh();}));
['checkpoint','x','metric','health-metric'].forEach(id=>$(id).addEventListener('change',refresh));
$('download').addEventListener('click',()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],
{type:'application/json'}));
const a=document.createElement('a');a.href=u;a.download='g1-learning-baselines-embedded.json';a.click();
setTimeout(()=>URL.revokeObjectURL(u),1000);});
$('inventory').textContent=runs.length+' recorded runs · '+data.sources.length+' immutable input(s)';
checkpointOptions();details();refresh();
</script></body></html>
"""


def main() -> None:
    """Render one or more recorded baseline manifests without altering inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measurements = load_measurements(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(measurements), encoding="utf-8")
    print(f"Report: {args.output.resolve()} ({len(measurements['runs'])} recorded runs)")


if __name__ == "__main__":
    main()
