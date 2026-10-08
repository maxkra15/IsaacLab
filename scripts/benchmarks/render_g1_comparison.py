# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render an interactive, offline G1 PPO comparison from recorded measurements.

Usage:
    uv run --no-sync python scripts/benchmarks/render_g1_comparison.py results.json comparison.html

The input contains ``metadata`` and ``runs``. Each run's ``evaluations`` is a
list of checkpoint measurements with ``scenario`` set to ``native_commands`` or
``forward_0_5``. Missing and nonfinite values are shown as unavailable.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import shlex
from pathlib import Path
from typing import Any


def _clean(value: Any) -> Any:
    """Replace nonfinite values before embedding the measurements in JavaScript."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def render_report(data: dict[str, Any]) -> str:
    """Create a portable HTML report with inline charts, interactions, and data."""
    data = _clean(data)
    metadata = data.get("metadata", {})
    runs = data.get("runs", [])
    for run in runs:
        selected_gpu = run.get("selected_gpu") or (metadata.get("selected_gpu") if len(runs) == 1 else {}) or {}
        run.setdefault("gpu_index", selected_gpu.get("index", run.get("device", "unknown")))
        run.setdefault("gpu_uuid", selected_gpu.get("uuid"))
        for point in run.get("evaluations", []):
            point.setdefault("linear_velocity_error_mean", point.get("velocity_xy_error_mean"))
            point.setdefault("yaw_velocity_error_mean", point.get("yaw_rate_error_mean"))
    title = html.escape(str(metadata.get("artifact_label", "G1 walking · PPO comparison")))
    hardware = html.escape(str(metadata.get("hardware", "Hardware details unavailable")))
    outcome = metadata.get("outcome_summary")
    outcome = f'<p class="outcome">{html.escape(str(outcome))}</p>' if outcome else ""
    details, pictures = [], []
    for run in runs:
        name = html.escape(str(run.get("name", run.get("algorithm", "Unnamed run"))))
        command = run.get("command", [])
        command = shlex.join(str(part) for part in command) if isinstance(command, list) else str(command)
        configuration = {
            key: value
            for key, value in run.items()
            if key not in ("command", "evaluations", "training_curve", "playback_image")
        }
        details.append(
            f"<details><summary>{name} · seed {html.escape(str(run.get('seed', '—')))}: command and configuration"
            f"</summary><pre>{html.escape(command)}</pre>"
            f"<pre>{html.escape(json.dumps(configuration, indent=2, ensure_ascii=False))}</pre></details>"
        )
        if run.get("playback_image"):
            picture = html.escape(str(run["playback_image"]), quote=True)
            pictures.append(
                f'<figure><img src="{picture}" alt="{name} policy playback"><figcaption>{name}</figcaption></figure>'
            )
    notes = []
    for key in ("notes", "evaluation_notes"):
        items = metadata.get(key, [])
        notes.extend([items] if isinstance(items, str) else items)
    notes = "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)
    payload = json.dumps(data, allow_nan=False, ensure_ascii=False).replace("<", "\\u003c")
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:light;--ink:#163047;--muted:#61748a;--line:#dce5ed;--warp:#087e74;--rsl:#416cba}
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1420px;padding:42px 30px;margin:auto}h1{font-size:40px;letter-spacing:-1.4px;line-height:1.15;
margin:10px 0 12px}h2{font-size:20px;margin:0 0 14px}p{margin:9px 0 16px}.eyebrow{color:var(--warp);
font-size:12px;letter-spacing:2px;font-weight:650}.muted,figcaption{color:var(--muted)}.small{font-size:12px}
.badge{display:inline-block;background:#e5eef5;padding:5px 10px;border-radius:20px;font-size:12px;margin:4px 6px 0 0}
.outcome{background:#eaf4f2;border-left:4px solid var(--warp);border-radius:8px;padding:14px 18px;
font-size:16px;margin:20px 0 0}.architecture{margin:16px 0}.architecture caption{text-align:left;font-weight:650;
padding-bottom:8px}.architecture td:first-child{width:160px}
.panel{border:1px solid var(--line);background:#fff;border-radius:16px;padding:24px;margin-top:22px;
box-shadow:0 5px 18px #17395404}.controls{display:flex;flex-wrap:wrap;gap:15px;align-items:center}
.controls label,.control-label{font-size:13px;color:var(--muted)}
select,button{font:inherit;border:1px solid var(--line);
background:white;border-radius:7px;padding:7px 10px;color:var(--ink)}select{margin-left:6px}button{cursor:pointer}
input{accent-color:var(--warp)}.legend label{margin-right:13px;white-space:nowrap}.control-row{margin-bottom:18px}
.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:13px;margin-top:20px}.card{padding:18px;
background:white;border:1px solid var(--line);border-radius:12px;min-height:132px}.card .value{font-size:27px;
font-weight:700;line-height:1.2;margin:8px 0}.card.good .value{color:var(--warp)}.card.bad .value{color:#ac623d}
.card-title{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}.card-note{font-size:12px;
color:var(--muted)}.scrubber{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:18px;margin:18px 0}
.scrubber input{width:100%}.grid{display:grid;grid-template-columns:1.5fr 1fr;gap:22px}.grid .panel{min-width:0}
.chart{display:block;width:100%;height:auto}.chart text{fill:var(--muted);font-size:12px}.scroll{overflow:auto}
table{border-collapse:collapse;width:100%;white-space:nowrap;text-align:left;font-variant-numeric:tabular-nums;font-size:13px}
th{padding:10px;border-bottom:2px solid var(--line);color:var(--muted);font-size:11px;text-transform:uppercase;
letter-spacing:.4px;cursor:pointer}th:focus{outline:2px solid var(--warp)}
td{padding:13px 10px;border-bottom:1px solid #edf1f5}
td:first-child{font-weight:600}tr:last-child td{border-bottom:0}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;
margin-right:8px}.seed-summary{display:flex;flex-wrap:wrap;gap:18px;font-size:13px;margin-top:15px}
.seed-summary span{padding:8px 12px;background:#f3f6f9;border-radius:7px}
details{padding:14px 0;border-top:1px solid var(--line)}summary{cursor:pointer;font-weight:600}
pre{background:#f3f6f9;border-radius:9px;padding:16px;font:12px/1.6 ui-monospace,monospace;
overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}a{color:var(--rsl)}
ul.notes{padding-left:20px}.notes li{margin:6px 0}
.tooltip{position:fixed;pointer-events:none;z-index:10;max-width:340px;background:#163047;color:#fff;border-radius:8px;
padding:12px 15px;font-size:12px;box-shadow:0 8px 25px #0002;white-space:pre-line;display:none}
.pictures{display:flex;gap:20px;flex-wrap:wrap}figure{flex:1;min-width:240px;margin:0}
figure img{width:100%;border-radius:9px}
footer{font-size:12px;color:var(--muted);margin-top:25px}@media(max-width:1100px){.cards{grid-template-columns:repeat(3,1fr)}
.grid{grid-template-columns:1fr}}@media(max-width:650px){main{padding:24px 14px}h1{font-size:31px}.panel{padding:18px}
.cards{grid-template-columns:1fr 1fr}.scrubber{grid-template-columns:1fr}
.controls{gap:12px}.card .value{font-size:24px}}
</style></head><body><main>
<header><div class="eyebrow">ISAAC LAB × ROBOLEARN × RSL-RL</div><h1>__TITLE__</h1>
<p class="muted">__HARDWARE__</p><span class="badge" id="budget"></span><span class="badge" id="replicas"></span>
<span class="badge">Recorded measurements · offline report</span>__OUTCOME__</header>
<section class="panel"><div class="controls control-row">
<div class="legend" id="algorithms" aria-label="Learner visibility"></div>
<label>Seed<select id="seed"><option value="all">All seeds</option></select></label>
<label>GPU<select id="gpu"><option value="all">All GPUs</option></select></label>
<label>Evaluation<select id="scenario"><option value="native_commands">Native velocity commands</option>
<option value="forward_0_5">Forward 0.5 m/s</option></select></label></div>
<div class="controls"><label>Horizontal axis<select id="x"><option value="time_seconds">Training time (s)</option>
<option value="transitions">Environment transitions</option>
<option value="iteration">Completed iterations</option></select></label>
<label>Quality metric<select id="metric"><option value="return_mean">Episode return ↑</option>
<option value="linear_velocity_error_mean">Planar velocity error ↓</option>
<option value="yaw_velocity_error_mean">Yaw-rate error ↓</option>
<option value="survival_rate">Survival ↑</option>
<option value="tracking_success_rate">Tracking success ↑</option>
<option value="walking_success_rate">Walking success (forward only) ↑</option>
<option value="forward_velocity_mean">Forward speed</option></select></label>
<label><input id="training" type="checkbox"> Training returns (dashed)</label>
<button id="download" type="button">Download measurements</button></div>
<div class="scrubber"><label for="checkpoint" class="control-label">Compare checkpoint budget</label>
<input id="checkpoint" type="range" min="0" value="0" step="1"><strong id="checkpoint-label"></strong></div>
<p class="small muted" id="selection-note">Cards and the table use the closest recorded checkpoint for each run.
The actual completed iteration is shown; checkpoint values are never interpolated.</p></section>
<div class="cards" id="cards"></div><p class="small muted" id="pair-note"></p>
<div class="grid"><section class="panel"><h2 id="chart-title">Quality over training</h2>
<svg id="learning" class="chart" viewBox="0 0 850 400" role="img"
aria-label="Measured policy quality over training"></svg>
<p class="small muted">Dots are deterministic checkpoint evaluations.
Lines connect recorded measurements for visual guidance;
they do not supply intermediate evaluations. Hover a dot for the seed, GPU, budget, and measurements.
Faint vertical markers identify the checkpoints selected by the budget slider.</p></section>
<section class="panel"><h2>Where training time goes</h2><svg id="phases" class="chart" viewBox="0 0 550 280"
role="img" aria-label="Measured rollout and update durations"></svg>
<p class="small muted">Blue: rollout, including actions and physics. Teal: learning update.
Phase timers synchronize CUDA.
Capture preparation is reported separately. Process wall includes the training process's launch, logging, checkpointing,
and shutdown; deterministic evaluation runs separately.</p></section></div>
<section class="panel"><h2>Selected checkpoint measurements</h2><div class="scroll"><table><thead><tr>
<th data-key="name" tabindex="0">Learner ↕</th><th data-key="seed" tabindex="0">Seed ↕</th>
<th data-key="gpu_index" tabindex="0">GPU ↕</th><th data-key="iteration" tabindex="0">Iteration ↕</th>
<th data-key="return_mean" tabindex="0">Return ↕</th>
<th data-key="linear_velocity_error_mean" tabindex="0">XY error (m/s) ↕</th>
<th data-key="yaw_velocity_error_mean" tabindex="0">Yaw error (rad/s) ↕</th>
<th data-key="survival_rate" tabindex="0">Survival ↕</th>
<th data-key="walking_success_rate" tabindex="0">Walking ↕</th>
<th data-key="forward_velocity_mean" tabindex="0">Forward (m/s) ↕</th>
<th data-key="episode_seconds_mean" tabindex="0">Episode (s) ↕</th>
<th data-key="training_seconds" tabindex="0">Train (s) ↕</th>
<th data-key="steady_fps" tabindex="0">Transitions/s ↕</th></tr></thead>
<tbody id="measurements"></tbody></table></div>
<div class="seed-summary" id="seed-summary"></div><p class="small muted">
Return ± episode standard deviation describes variation across evaluation episodes.
Where two or more distinct training seeds are available, the summary reports their mean ± sample standard deviation.
This is observed seed spread, not a confidence interval.
Two seeds cannot establish a stable algorithm ranking.</p></section>
<section class="panel"><h2>Protocol and interpretation</h2>
<p>Both learners use the recorded G1 MDP, physics backend, observation/action contract, and collection budget.
Planar velocity error is the mean Euclidean error in the yaw frame (m/s), and yaw-rate error is the mean absolute
world-z error (rad/s). Survival is the fraction of evaluation episodes reaching timeout without a failure termination.
Low tracking error should be considered together with survival and return.</p>
<div class="scroll"><table class="architecture" aria-label="Shared simulation and learner execution paths">
<caption>Execution path</caption><thead><tr><th>Stage</th><th>RSL-RL PPO</th><th>RoboLearn Warp PPO</th></tr></thead>
<tbody><tr><td>Physics</td><td colspan="2">Shared Newton/MuJoCo Warp physics graph</td></tr>
<tr><td>Isaac Lab MDP</td><td colspan="2">
Shared eager Torch observations, rewards, resets, and action processing</td></tr>
<tr><td>PPO update</td><td>Torch learner</td><td>Warp-NN learner captured in a CUDA graph</td></tr>
</tbody></table></div>
<p>Warp PPO captures the learner update in a CUDA graph. The Isaac Lab rollout remains eager.
The learner graph is separate from physics stepping; this run does not capture the complete rollout and learning cycle
in one graph.
Speed ratios compare measured implementations and configurations; quality cards can favor either learner.
Checkpoint evaluation uses deterministic actions and the recorded reset seeds and command scenarios.
Unavailable measurements are shown as —.</p><ul class="notes">__NOTES__</ul>
__DETAILS__<details><summary>Full experiment metadata</summary><pre>__METADATA__</pre></details></section>
<section class="panel" id="playback"><h2>Policy playback</h2><div class="pictures">__PICTURES__</div></section>
<footer>PPO: <a href="https://arxiv.org/abs/1707.06347">John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford,
and Oleg Klimov (2017)</a>. Implementations: <a href="https://github.com/leggedrobotics/rsl_rl">RSL-RL</a>,
<a href="https://github.com/maxkra15/RoboLearn">RoboLearn</a>, and NVIDIA <a href="https://nvidia.github.io/warp-nn/">Warp-NN</a>.
Learning-curve presentation follows return versus time and samples in <a href="https://arxiv.org/html/2604.04539v1#S12">FlashSAC</a>.
All data and interactions are embedded in this file; no network access is needed.</footer>
</main><div class="tooltip" id="tooltip"></div><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),runs=data.runs||[],meta=data.metadata||{};
const $=id=>document.getElementById(id),finite=x=>typeof x==='number'&&Number.isFinite(x);
const metrics={return_mean:{label:'Mean episode return',unit:'return',scale:1,higher:true},
linear_velocity_error_mean:{label:'Planar velocity error (m/s)',unit:'m/s',scale:1,higher:false},
yaw_velocity_error_mean:{label:'Yaw-rate error (rad/s)',unit:'rad/s',scale:1,higher:false},
survival_rate:{label:'Survival (%)',unit:'percentage points',scale:100,higher:true},
tracking_success_rate:{label:'Tracking success (%)',unit:'percentage points',scale:100,higher:true},
walking_success_rate:{label:'Walking success (%)',unit:'percentage points',scale:100,higher:true},
forward_velocity_mean:{label:'Forward speed (m/s)',unit:'m/s',scale:1,higher:null}};
const axes={time_seconds:'Training time (s)',transitions:'Environment transitions',iteration:'Completed iterations'};
const algorithms=[...new Set(runs.map(r=>r.algorithm||r.name||'unknown'))];
const enabled=new Set(algorithms),palette=['#087e74','#416cba','#bd7840','#9a66b7'];
const color=r=>palette[algorithms.indexOf(r.algorithm||r.name||'unknown')%palette.length];
const gpu=r=>String(r.gpu_index??r.gpu_uuid??'unknown');
const fmt=(v,d=2)=>finite(v)?v.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
const compact=v=>Math.abs(v)>=1e6?(v/1e6).toFixed(1)+'M':
Math.abs(v)>=1e3?(v/1e3).toFixed(1)+'k':fmt(v,Math.abs(v)<10?2:0);
const mean=a=>a.reduce((sum,value)=>sum+value,0)/a.length;
const sd=a=>a.length>1?Math.sqrt(a.reduce((sum,value)=>sum+(value-mean(a))**2,0)/(a.length-1)):null;
const scenarioName=value=>!value||value==='native'?'native_commands':value;
const scaled=(value,scale)=>finite(value)?value*scale:null;
const values=(run,scenario=$('scenario').value)=>(run.evaluations||[]).filter(p=>scenarioName(p.scenario)===scenario);
function visible(){return runs.filter(r=>enabled.has(r.algorithm||r.name||'unknown')&&
($('seed').value==='all'||String(r.seed)===$('seed').value)&&($('gpu').value==='all'||gpu(r)===$('gpu').value));}
function selected(run){const target=Number($('checkpoint').value);
return values(run).filter(p=>finite(p.iteration)).sort((a,b)=>
Math.abs(a.iteration-target)-Math.abs(b.iteration-target)||a.iteration-b.iteration)[0]||{};}
function append(parent,tag,text,className){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;
if(className)el.className=className;parent.appendChild(el);return el;}
const ns='http://www.w3.org/2000/svg';
function svgElement(parent,tag,attributes,text){const el=document.createElementNS(ns,tag);
Object.entries(attributes).forEach(([key,value])=>el.setAttribute(key,value));
if(text!==undefined)el.textContent=text;parent.appendChild(el);return el;}
function tooltip(event,text){const box=$('tooltip');box.textContent=text;box.style.display='block';
box.style.left=Math.min(event.clientX+14,window.innerWidth-box.offsetWidth-10)+'px';
box.style.top=Math.min(event.clientY+14,window.innerHeight-box.offsetHeight-10)+'px';}
function chart(){const svg=$('learning');svg.replaceChildren();
const metric=$('metric').value,xkey=$('x').value,spec=metrics[metric];
const showTraining=$('training').checked&&metric==='return_mean';const series=[];
visible().forEach((run,index)=>{const points=values(run).filter(p=>finite(p[xkey])&&finite(p[metric]))
.sort((a,b)=>a[xkey]-b[xkey]);
series.push({run,index,points,training:false});if(showTraining)series.push({run,index,
points:(run.training_curve||[]).filter(p=>finite(p[xkey])&&finite(p.return_mean))
.sort((a,b)=>a[xkey]-b[xkey]),training:true});});
const points=series.flatMap(s=>s.points);$('chart-title').textContent=spec.label+' vs '+axes[xkey].toLowerCase();
if(!points.length){svgElement(svg,'text',{x:45,y:90},'No measurements for the current selection');return;}
const xmax=Math.max(1,...points.map(p=>p[xkey])),raw=points.map(p=>p[metric]*spec.scale);
const percentage=spec.scale===100,low=Math.min(0,...raw),high=Math.max(percentage?100:1,...raw);
const pad=Math.max(.01,(high-low)*.06),ymin=percentage?0:low-pad,ymax=percentage?100:high+pad;
const x=v=>75+v/xmax*740,y=v=>340-(v-ymin)/(ymax-ymin)*285;
for(let i=0;i<=5;i++){const a=i/5*xmax,b=ymin+i/5*(ymax-ymin);
svgElement(svg,'line',{x1:75,y1:y(b),x2:815,y2:y(b),stroke:'#e6edf3'});
svgElement(svg,'text',{x:65,y:y(b)+4,'text-anchor':'end'},compact(b));
svgElement(svg,'text',{x:x(a),y:365,'text-anchor':'middle'},compact(a));}
svgElement(svg,'text',{x:75,y:25},spec.label);svgElement(svg,'text',{x:440,y:390,'text-anchor':'middle'},axes[xkey]);
series.forEach(s=>{const shade=color(s.run),seedDash=s.index%3===0?'none':s.index%3===1?'8 3':'3 3';
svgElement(svg,'polyline',{points:s.points.map(p=>`${x(p[xkey])},${y(p[metric]*spec.scale)}`).join(' '),
fill:'none',stroke:shade,'stroke-width':s.training?1.2:2.4,
'stroke-dasharray':s.training?'5 5':seedDash,opacity:s.training?.28:.82});
if(s.training)return;const chosen=selected(s.run);
if(finite(chosen[xkey]))svgElement(svg,'line',{x1:x(chosen[xkey]),y1:55,x2:x(chosen[xkey]),y2:340,
stroke:shade,'stroke-dasharray':'3 5',opacity:.23});
s.points.forEach(p=>{const dot=svgElement(svg,'circle',{cx:x(p[xkey]),cy:y(p[metric]*spec.scale),
r:p===chosen?6:4.5,fill:shade,stroke:'white','stroke-width':1.5,tabindex:0});
const description=`${s.run.name||s.run.algorithm} · seed ${s.run.seed??'—'} · GPU ${gpu(s.run)}\n`+
`Iteration ${fmt(p.iteration,0)} · ${fmt(p.transitions,0)} transitions\nTraining ${fmt(p.time_seconds)} s\n`+
`Return ${fmt(p.return_mean)} ± ${fmt(p.return_std)} episode SD\nXY error ${fmt(p.linear_velocity_error_mean,3)} m/s\n`+
`Yaw error ${fmt(p.yaw_velocity_error_mean,3)} rad/s\nSurvival ${fmt(scaled(p.survival_rate,100),1)}%\n`+
`Forward speed ${fmt(p.forward_velocity_mean,3)} m/s\nWalking success ${fmt(scaled(p.walking_success_rate,100),1)}%`;
dot.onpointermove=event=>tooltip(event,description);dot.onpointerleave=()=>{$('tooltip').style.display='none';};
svgElement(dot,'title',{},description);});});}
function pairs(){const available=visible(),warps=available.filter(r=>r.algorithm==='warp_ppo'),
baselines=available.filter(r=>['rsl_rl_ppo','rsl_rl'].includes(r.algorithm));const paired=[],used=new Set();
warps.forEach(warp=>{const candidates=baselines.filter(r=>r.seed===warp.seed&&!used.has(r));
const baseline=candidates.find(r=>gpu(r)===gpu(warp))||candidates[0];
if(baseline){paired.push({warp,baseline});used.add(baseline);}});return paired;}
function card(title,value,note,good=null){const el=append($('cards'),'div',undefined,
'card'+(good===true?' good':good===false?' bad':''));
append(el,'div',title,'card-title');append(el,'div',value,'value');append(el,'div',note,'card-note');}
function cards(){const matched=pairs();$('cards').replaceChildren();
[['Training loop','training_seconds'],['Learning update','update_seconds'],['Rollout','rollout_seconds'],
['Process wall','process_wall_seconds']]
.forEach(([title,key])=>{const ratios=matched.filter(p=>finite(p.baseline[key])&&p.baseline[key]>0&&
finite(p.warp[key])&&p.warp[key]>0)
.map(p=>p.baseline[key]/p.warp[key]);if(!ratios.length){card(title,'—','No comparable pair selected');return;}
const ratio=mean(ratios),spread=sd(ratios);card(title,fmt(ratio,2)+'×',
(ratio===1?'Equal time':ratio>1?'Warp faster':'Warp slower')+' · RSL time / Warp time'+
(spread!==null?' · SD '+fmt(spread,2):''),ratio===1?null:ratio>1);});
const metric=$('metric').value,spec=metrics[metric],differences=matched.map(p=>{
const a=selected(p.baseline)[metric],b=selected(p.warp)[metric];
return finite(a)&&finite(b)?(b-a)*spec.scale:null;}).filter(finite);
if(!differences.length)card('Selected policy quality','—','No comparable evaluations selected');
else{const delta=mean(differences),better=spec.higher?delta>0:delta<0,spread=sd(differences);
const judgment=spec.higher===null?(delta>0?'Warp higher':'Warp lower'):better?'Warp better':'Warp worse';
card('Selected policy quality',(delta>0?'+':'')+fmt(delta,spec.scale===100?1:3),
'Warp − RSL '+spec.unit+' · '+(delta===0?'equal':judgment)+
(spread!==null?' · SD '+fmt(spread,3):''),delta===0||spec.higher===null?null:better);}
const crossGpu=matched.some(p=>gpu(p.warp)!==gpu(p.baseline));$('pair-note').textContent=matched.length?
`Ratios average ${matched.length} paired run${matched.length===1?'':'s'} with matching training seeds. `+
(crossGpu?'Some pairs use different GPU devices; interpret hardware differences alongside the results. ':
'Pairs use the same GPU device. ')+
'Speed ratios use full training-run totals. Quality uses the closest recorded checkpoints at the selected budget.':
'Select both learners with a matching seed to compare measured speed and quality. No advantage is assumed.';}
function phases(){const svg=$('phases');svg.replaceChildren();
const current=visible(),height=Math.max(240,current.length*50+70);svg.setAttribute('viewBox',`0 0 550 ${height}`);
const max=Math.max(1,...current.map(r=>(r.rollout_seconds||0)+(r.update_seconds||0)));
if(!current.length){svgElement(svg,'text',{x:20,y:70},'No runs selected');return;}
current.forEach((run,index)=>{const top=28+index*50;let left=170;
svgElement(svg,'text',{x:0,y:top+18},`${run.name||run.algorithm} · s${run.seed??'—'} · g${gpu(run)}`);
[['rollout_seconds','#7395c8'],['update_seconds','#087e74']].forEach(([key,shade])=>{if(!finite(run[key]))return;
const width=run[key]/max*290,bar=svgElement(svg,'rect',{x:left,y:top,width,height:25,rx:3,fill:shade});
svgElement(bar,'title',{},`${key}: ${fmt(run[key],3)} s`);left+=width;});
svgElement(svg,'text',{x:left+7,y:top+18},finite(run.training_seconds)?fmt(run.training_seconds,1)+' s':'—');});
svgElement(svg,'text',{x:170,y:height-12},'Rollout + learner update (s)');}
let sortKey='name',sortDirection=1;
function table(){const rows=visible().map(run=>({run,point:selected(run)}));
const value=(row,key)=>key in row.point?row.point[key]:row.run[key];
rows.sort((a,b)=>{const x=value(a,sortKey),y=value(b,sortKey);if(x==null)return y==null?0:1;if(y==null)return -1;
return sortDirection*(finite(x)&&finite(y)?x-y:String(x).localeCompare(String(y),undefined,{numeric:true}));});
$('measurements').replaceChildren();rows.forEach(({run,point})=>{const row=append($('measurements'),'tr');
const name=append(row,'td');const dot=append(name,'span',undefined,'dot');dot.style.background=color(run);
name.appendChild(document.createTextNode(run.name||run.algorithm));
[run.seed??'—',gpu(run),fmt(point.iteration,0),fmt(point.return_mean)+' ± '+fmt(point.return_std),
fmt(point.linear_velocity_error_mean,3),fmt(point.yaw_velocity_error_mean,3),fmt(scaled(point.survival_rate,100),1)+'%',
fmt(scaled(point.walking_success_rate,100),1)+'%',fmt(point.forward_velocity_mean,3),fmt(point.episode_seconds_mean,1),
fmt(run.training_seconds,1),fmt(run.steady_fps,0)].forEach(item=>append(row,'td',item));});
if(!rows.length){const cell=append(append($('measurements'),'tr'),'td','No runs selected');cell.colSpan=13;}
$('seed-summary').replaceChildren();const metric=$('metric').value,spec=metrics[metric];
algorithms.forEach(algorithm=>{const entries=rows.filter(r=>(r.run.algorithm||r.run.name)===algorithm&&
finite(r.point[metric]));
const seeds=new Set(entries.map(r=>r.run.seed));if(seeds.size<2)return;
const bySeed=[...seeds].map(seed=>mean(entries.filter(r=>r.run.seed===seed).map(r=>r.point[metric]*spec.scale)));
append($('seed-summary'),'span',`${entries[0].run.name||algorithm}: ${fmt(mean(bySeed),3)} ± ${fmt(sd(bySeed),3)} `+
`${spec.unit} · ${seeds.size} training seeds`);});}
function redraw(){const max=Math.max(0,
...runs.flatMap(r=>(r.evaluations||[]).map(p=>finite(p.iteration)?p.iteration:0)),meta.iterations||0);
$('checkpoint').max=max;$('checkpoint-label').textContent='Iteration '+fmt(Number($('checkpoint').value),0);
const forward=$('scenario').value==='forward_0_5';
$('metric').querySelector('option[value="walking_success_rate"]').disabled=!forward;
if(!forward&&$('metric').value==='walking_success_rate')$('metric').value='tracking_success_rate';
$('training').disabled=$('metric').value!=='return_mean';chart();cards();phases();table();}
algorithms.forEach(algorithm=>{const run=runs.find(r=>(r.algorithm||r.name||'unknown')===algorithm),
label=append($('algorithms'),'label');
label.style.color=color(run);const input=append(label,'input');input.type='checkbox';input.checked=true;
input.onchange=()=>{input.checked?enabled.add(algorithm):enabled.delete(algorithm);redraw();};
label.appendChild(document.createTextNode(' '+(run.name||algorithm)));});
function options(id,items,label){[...new Set(items)]
.sort((a,b)=>String(a).localeCompare(String(b),undefined,{numeric:true}))
.forEach(value=>{const option=append($(id),'option',label(value));option.value=String(value);});}
options('seed',runs.map(r=>r.seed??'unknown'),x=>'Seed '+x);options('gpu',runs.map(gpu),x=>'GPU '+x);
const scenarios=new Set(runs.flatMap(r=>(r.evaluations||[]).map(p=>scenarioName(p.scenario))));
if(scenarios.size&&!scenarios.has('native_commands'))$('scenario').value=[...scenarios][0];
const maxIteration=Math.max(0,meta.iterations||0,
...runs.flatMap(r=>(r.evaluations||[]).map(p=>finite(p.iteration)?p.iteration:0)));
$('checkpoint').max=maxIteration;$('checkpoint').value=maxIteration;
['seed','gpu','scenario','x','metric','training'].forEach(id=>$(id).onchange=redraw);$('checkpoint').oninput=redraw;
document.querySelectorAll('th[data-key]').forEach(header=>{const sort=()=>{
sortDirection=sortKey===header.dataset.key?-sortDirection:1;sortKey=header.dataset.key;table();};
header.onclick=sort;header.onkeydown=event=>{
if(event.key==='Enter'||event.key===' '){event.preventDefault();sort();}};});
$('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],
{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='g1-comparison.json';
link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('budget').textContent=`${fmt(meta.num_envs??runs[0]?.num_envs,0)} environments × `+
`${fmt(meta.horizon??runs[0]?.horizon,0)} steps × ${fmt(meta.iterations??runs[0]?.iterations,0)} iterations per run`;
$('replicas').textContent=`${new Set(runs.map(r=>r.seed)).size} training seed(s) · `+
`${new Set(runs.map(gpu)).size} GPU device(s)`;
if(!document.querySelector('.pictures img'))$('playback').remove();redraw();
</script></body></html>"""
    replacements = {
        "TITLE": title,
        "HARDWARE": hardware,
        "OUTCOME": outcome,
        "NOTES": notes,
        "DETAILS": "".join(details),
        "METADATA": html.escape(json.dumps(metadata, indent=2, ensure_ascii=False)),
        "PICTURES": "".join(pictures),
        "DATA": payload,
    }
    for key, value in replacements.items():
        template = template.replace(f"__{key}__", value)
    return template


def main() -> None:
    """Read the measurements and write a self-contained HTML artifact."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Recorded comparison measurements as JSON.")
    parser.add_argument("output", type=Path, help="Destination HTML file.")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    for run in data.get("runs", []):
        picture = run.get("playback_image")
        if picture and not picture.startswith("data:"):
            image_path = args.input.parent / picture
            if image_path.suffix.lower() == ".png" and image_path.is_file():
                run["playback_image"] = "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode(
                    "ascii"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(data), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
