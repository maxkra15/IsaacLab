# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render an offline Cartpole learner comparison from recorded experiment JSON.

Usage:
    uv run python scripts/benchmarks/render_robolearn_comparison.py results.json comparison.html

The input contains ``metadata`` and ``runs``. Missing or nonfinite measurements
are displayed as unavailable. The renderer does not run or import a simulator.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shlex
from pathlib import Path
from typing import Any


def _clean(value: Any) -> Any:
    """Replace nonfinite JSON numbers with null before embedding in JavaScript."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _format(value: Any, decimals: int = 2, suffix: str = "", scale: float = 1.0) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "—"
    return f"{value * scale:,.{decimals}f}{suffix}"


def render_report(data: dict[str, Any]) -> str:
    """Create a self-contained report without external scripts, fonts, or assets."""
    data = _clean(data)
    metadata, runs = data.get("metadata", {}), data.get("runs", [])
    rows, details, thresholds, pictures = [], [], [], []
    for run in runs:
        name = html.escape(str(run.get("name", "Unnamed run")))
        evaluations = run.get("evaluations") or []
        final = evaluations[-1] if evaluations else {}
        values = [
            name,
            _format(final.get("return_mean")),
            _format(final.get("normalized_score"), 1, "%"),
            _format(final.get("episode_length_mean"), 1),
            _format(final.get("survival_rate"), 1, "%", scale=100),
            _format(final.get("upright_fraction"), 1, "%", scale=100),
            _format(run.get("training_seconds"), 2, " s"),
            _format(run.get("steady_fps"), 0),
            _format(run.get("process_wall_seconds"), 2, " s"),
            _format(run.get("warmup_seconds"), 2, " s"),
        ]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
        reached = next(
            (
                point
                for point in evaluations
                if isinstance(point.get("normalized_score"), (int, float)) and point["normalized_score"] >= 90.0
            ),
            None,
        )
        threshold = "Not reached at a recorded evaluation"
        if reached:
            threshold = (
                f"{_format(reached.get('time_seconds'), 2, ' s')} · "
                f"{_format(reached.get('transitions'), 0)} transitions · "
                f"iteration {_format(reached.get('iteration'), 0)}"
            )
        thresholds.append(f"<li><strong>{name}</strong><span>{threshold}</span></li>")
        command = run.get("command", [])
        command = shlex.join(str(part) for part in command) if isinstance(command, list) else str(command)
        fields = {
            key: run.get(key)
            for key in (
                "algorithm",
                "iterations",
                "num_envs",
                "horizon",
                "transitions",
                "parameter_count",
                "actor_gradient_updates",
                "critic_gradient_updates",
                "peak_torch_memory_mib",
                "agent_config",
                "algorithm_config",
                "log_dir",
            )
        }
        details.append(
            f"<details><summary>{name}: command and run configuration</summary>"
            f"<pre>{html.escape(command)}</pre><pre>{html.escape(json.dumps(fields, indent=2))}</pre></details>"
        )
        if run.get("playback_image"):
            path = html.escape(str(run["playback_image"]), quote=True)
            pictures.append(
                f'<figure><a href="{path}"><img src="{path}" alt="{name} playback"></a>'
                f"<figcaption>{name} · playback on DISPLAY=:1</figcaption></figure>"
            )
    title = html.escape(str(metadata.get("artifact_label", "Cartpole · RoboLearn comparison")))
    hardware = html.escape(str(metadata.get("hardware", "Hardware details unavailable")))
    notes = html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))
    payload = json.dumps(data, allow_nan=False, ensure_ascii=False).replace("<", "\\u003c")
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:light;--ink:#14263b;--muted:#607185;--border:#dce5ed;--blue:#176aaf}
*{box-sizing:border-box}body{margin:0;background:#f5f8fb;color:var(--ink);font:15px/1.6 system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:46px 30px}h1{font-size:38px;letter-spacing:-1.5px;margin:6px 0 8px}
h2{font-size:20px;margin:0 0 15px}p{margin:8px 0 16px}.eyebrow{font-size:12px;letter-spacing:2px;color:var(--blue)}
.muted,figcaption{color:var(--muted)}.badge{display:inline-block;padding:4px 11px;border-radius:20px;
background:#e5eff7;color:#345a76;font-size:12px;margin-right:7px}.panel{background:white;border:1px solid var(--border);
border-radius:15px;padding:24px;margin-top:22px;box-shadow:0 5px 20px #17395405}.grid{display:grid;
grid-template-columns:1fr 1fr;gap:22px}.grid .panel{min-width:0}.scroll{overflow:auto}table{width:100%;
border-collapse:collapse;text-align:left;font-variant-numeric:tabular-nums;white-space:nowrap;font-size:13px}
th{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);padding:10px 12px;
border-bottom:2px solid var(--border)}td{padding:14px 12px;border-bottom:1px solid #edf1f5}
td:first-child{font-weight:650}tr:last-child td{border-bottom:0}.chart{width:100%;height:auto;display:block}
.chart text{fill:var(--muted);font-size:11px}
.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:10px 0;color:var(--muted);font-size:13px}
input{accent-color:var(--blue)}label{cursor:pointer}details{border-top:1px solid var(--border);padding:14px 0}
summary{cursor:pointer;font-weight:600}pre{background:#f3f6f9;padding:16px;border-radius:8px;overflow:auto;
font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}a{color:var(--blue)}
.thresholds{padding:0;list-style:none}.thresholds li{display:flex;gap:20px;justify-content:space-between;
padding:11px 0;border-bottom:1px solid #edf1f5}.thresholds span{color:var(--muted)}.small{font-size:12px}
.pictures{display:flex;gap:20px;flex-wrap:wrap}figure{margin:0;flex:1;min-width:220px}img{width:100%;border-radius:9px}
footer{margin-top:28px;color:var(--muted);font-size:12px}@media(max-width:850px){.grid{grid-template-columns:1fr}
main{padding:25px 16px}h1{font-size:30px}.panel{padding:18px}.thresholds li{display:block}}
</style></head><body><main>
<header><div class="eyebrow">ISAAC LAB × ROBOLEARN</div><h1>__TITLE__</h1>
<p class="muted">__HARDWARE__</p><span class="badge">One seed · quick comparison</span>
<span class="badge">Same Cartpole MDP</span><span class="badge">Offline report</span></header>
<section class="panel"><h2>Outcome and training speed</h2><div class="scroll"><table><thead><tr>
<th>Learner</th><th>Final return</th><th>Score</th><th>Episode steps</th><th>Survival</th><th>Upright</th>
<th>Training</th><th>Transitions/s</th>
<th>Process wall</th><th>Warmup</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="small muted">Final values come from recorded deterministic evaluation. Score = 100 × return / 5:
five seconds of perfect reward is the theoretical upper bound for this manager MDP. Negative scores remain visible.
Survival is the fraction reaching the episode timeout; upright is the fraction of episode steps with
|pole angle| &lt; 0.2 rad. This normalization differs from the FlashSAC paper's asymptotic reference.
Process wall time covers the training subprocess, including startup, logging, checkpointing, and shutdown.
Evaluation runs separately. Training time sums synchronized rollout and update timers; Warp preparation is recorded
as warmup. Steady throughput omits the first five training iterations. — means unavailable.</p></section>
<div class="controls" id="legend"></div><div class="controls"><label><input id="training" type="checkbox">
Show training episode curves (dashed)</label><span>Dots: evaluations · hover for measurements</span></div>
<div class="grid"><section class="panel"><h2>Compute efficiency</h2><svg id="time" class="chart"
viewBox="0 0 550 330" role="img" aria-label="Episode return against training time"></svg></section>
<section class="panel"><h2>Sample efficiency</h2><svg id="samples" class="chart" viewBox="0 0 550 330"
role="img" aria-label="Episode return against environment transitions"></svg></section></div>
<div class="grid"><section class="panel"><h2>Where training time goes</h2><svg id="phases" class="chart"
viewBox="0 0 550 230" role="img" aria-label="Rollout and learner update time"></svg>
<p class="small muted">Rollout includes action selection and simulator interaction. Update time includes learner work.
Warmup/capture and evaluations are separate. Only recorded phases are drawn.</p></section>
<section class="panel"><h2>First measured score ≥90%</h2><ul class="thresholds">__THRESHOLDS__</ul>
<p class="small muted">No interpolation between evaluation checkpoints. The budget contains the same number of
16-step collection iterations for all learners; replay updates and PPO epochs have different compute costs.</p>
</section></div>
<section class="panel"><h2>Interpretation and reproducibility</h2>
<p>All learners use the same observations, resets, rewards, action bounds, timestep, and physics backend.
Evaluation reports raw environment rewards. The PPO comparison uses matching network sizes, rollout horizons,
full-batch epochs, and fixed learning rates. Warp PPO clips the combined gradient norm; RSL-RL clips actor and
critic norms separately. RoboLearn bootstraps timeouts from the final observation; native RSL-RL PPO uses the
current-state value. These are learner implementation differences; the environment MDP stays the same.
Warp PPO captures the learning update in a CUDA graph; Isaac Lab rollout assembly remains eager.
PPO parameter initialization also differs across implementations.</p>
<p class="muted">This is a single training seed on shared GPU hardware. Evaluation episodes quantify performance
within this run; they do not supply uncertainty across training seeds. These results describe this configuration
and budget, and do not establish an algorithm ranking. Survival alone does not establish pole balance.</p>
__DETAILS__<details><summary>Experiment metadata and protocol</summary><pre>__NOTES__</pre></details></section>
<section class="panel" id="playback"><h2>Policy playback</h2><div class="pictures">__PICTURES__</div></section>
<footer>Metrics follow the return-versus-time and return-versus-samples presentation in
<a href="https://arxiv.org/html/2604.04539v1#S12">FlashSAC</a>, Donghu Kim et al. (2026).
PPO: <a href="https://arxiv.org/abs/1707.06347">John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, Oleg Klimov (2017)</a>. Implementations:
<a href="https://github.com/maxkra15/RoboLearn">RoboLearn</a> and
<a href="https://github.com/leggedrobotics/rsl_rl">RSL-RL</a>.
Warp learning uses NVIDIA's <a href="https://nvidia.github.io/warp-nn/">Warp-NN</a>.
All plots use measured JSON values; no remote resources are required to view this report.</footer>
</main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.querySelector('#data').textContent), runs=data.runs||[];
const colors=['#176aaf','#bc6b2c','#008a78','#9064bb'], enabled=runs.map(()=>true);
const finite=x=>typeof x==='number'&&Number.isFinite(x), ns='http://www.w3.org/2000/svg';
function element(svg,tag,attrs,text){const el=document.createElementNS(ns,tag);
Object.entries(attrs).forEach(([key,value])=>el.setAttribute(key,value));
if(text!==undefined)el.textContent=text;svg.appendChild(el);return el;}
function number(x){return Math.abs(x)>=1e6?(x/1e6).toFixed(1)+'M':
Math.abs(x)>=1e3?(x/1e3).toFixed(1)+'k':x.toFixed(Math.abs(x)<10?1:0);}
function chart(id,key,label){const svg=document.getElementById(id);svg.replaceChildren();
const series=[];runs.forEach((run,i)=>{if(!enabled[i])return;
series.push({i,run,points:run.evaluations||[],train:false});
if(document.getElementById('training').checked)series.push({i,run,points:run.training_curve||[],train:true});});
series.forEach(s=>s.points=s.points.filter(p=>finite(p[key])&&finite(p.return_mean)));
const points=series.flatMap(s=>s.points);
if(!points.length){element(svg,'text',{x:50,y:80},'No measurements available');return;}
const xmax=Math.max(1,...points.map(p=>p[key])), low=Math.min(0,...points.map(p=>p.return_mean));
const high=Math.max(5,...points.map(p=>p.return_mean)), pad=Math.max(.2,(high-low)*.07);
const ymin=low-pad,ymax=high+pad,x=v=>65+v/xmax*465,y=v=>280-(v-ymin)/(ymax-ymin)*245;
for(let i=0;i<=4;i++){const a=i/4*xmax,b=ymin+i/4*(ymax-ymin);
element(svg,'line',{x1:65,y1:y(b),x2:530,y2:y(b),stroke:'#e8eef3'});
element(svg,'text',{x:54,y:y(b)+4,'text-anchor':'end'},number(b));
element(svg,'text',{x:x(a),y:300,'text-anchor':'middle'},number(a));}
element(svg,'text',{x:295,y:325,'text-anchor':'middle'},label);
element(svg,'text',{x:65,y:16},'Mean episode return');
element(svg,'line',{x1:65,y1:y(5),x2:530,y2:y(5),stroke:'#b1bbc5','stroke-dasharray':'4 5'});
series.forEach(s=>{const color=colors[s.i%colors.length];
element(svg,'polyline',{points:s.points.map(p=>`${x(p[key])},${y(p.return_mean)}`).join(' '),
fill:'none',stroke:color,'stroke-width':s.train?1.5:2.5,'stroke-dasharray':s.train?'5 4':'none',opacity:s.train?.45:1});
if(!s.train)s.points.forEach(p=>{const circle=element(svg,'circle',{cx:x(p[key]),cy:y(p.return_mean),r:4,fill:color});
element(circle,'title',{},`${s.run.name}: iteration ${p.iteration??'—'}\nReturn ${p.return_mean.toFixed(3)}`+
`\nTime ${finite(p.time_seconds)?p.time_seconds.toFixed(2)+' s':'—'}`+
`\nTransitions ${finite(p.transitions)?p.transitions.toLocaleString():'—'}`+
`\nEpisode return std ${finite(p.return_std)?p.return_std.toFixed(3):'—'} (not a training-seed CI)`);});});}
function phases(){const svg=document.getElementById('phases');svg.replaceChildren();
const available=runs.filter(r=>finite(r.rollout_seconds)||finite(r.update_seconds));
if(!available.length){element(svg,'text',{x:20,y:60},'No phase timings available');return;}
const max=Math.max(1,...available.map(r=>(finite(r.rollout_seconds)?r.rollout_seconds:0)+
(finite(r.update_seconds)?r.update_seconds:0))), height=Math.max(230,runs.length*45+55);
svg.setAttribute('viewBox',`0 0 550 ${height}`);
runs.forEach((run,i)=>{const top=30+i*45;let left=110;
element(svg,'text',{x:0,y:top+18},run.name);
[['rollout_seconds','#426f95'],['update_seconds','#e1b987']].forEach(([key,color])=>{
if(!finite(run[key]))return;const width=run[key]/max*360;
const rect=element(svg,'rect',{x:left,y:top,width,height:25,fill:color,rx:3});
element(rect,'title',{},`${key}: ${run[key].toFixed(3)} s`);left+=width;});
const total=finite(run.rollout_seconds)&&finite(run.update_seconds)?run.rollout_seconds+run.update_seconds:null;
element(svg,'text',{x:left+8,y:top+18},finite(total)?total.toFixed(2)+' s':'partial');});
element(svg,'text',{x:110,y:height-15},'Blue: rollout · amber: updates');}
function redraw(){chart('time','time_seconds','Training time (s)');
chart('samples','transitions','Environment transitions');}
runs.forEach((run,i)=>{const label=document.createElement('label'),input=document.createElement('input');
input.type='checkbox';input.checked=true;input.onchange=()=>{enabled[i]=input.checked;redraw();};
label.style.color=colors[i%colors.length];label.append(input,document.createTextNode(' '+run.name));
document.getElementById('legend').appendChild(label);});
document.getElementById('training').onchange=redraw;redraw();phases();
if(!document.querySelector('.pictures img'))document.getElementById('playback').remove();
</script></body></html>"""
    for key, value in {
        "TITLE": title,
        "HARDWARE": hardware,
        "ROWS": "".join(rows),
        "THRESHOLDS": "".join(thresholds),
        "DETAILS": "".join(details),
        "NOTES": notes,
        "PICTURES": "".join(pictures),
        "DATA": payload,
    }.items():
        template = template.replace(f"__{key}__", value)
    return template


def main() -> None:
    """Read recorded measurements and write a portable HTML artifact."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Comparison measurements as JSON.")
    parser.add_argument("output", type=Path, help="Destination HTML file.")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(json.loads(args.input.read_text())), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
