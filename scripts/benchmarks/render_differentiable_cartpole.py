# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render an offline report for the native physics-to-policy Cartpole experiment.

Usage:
    uv run --no-project python scripts/benchmarks/render_differentiable_cartpole.py \
        --input logs/pathwise/results.json --output logs/pathwise/report.html

The sibling ``metrics.jsonl`` is embedded when present. No simulator, plotting
library, network connection, or JavaScript dependency is needed to view the report.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


def _clean(value: Any) -> Any:
    """Replace non-finite diagnostic values with explicit missing measurements."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def render_report(result: dict[str, Any], metrics: list[dict[str, Any]] | None = None) -> str:
    """Render recorded gradients, training diagnostics, and native evaluations."""
    payload = {"result": _clean(result), "metrics": _clean(metrics or [])}
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False).replace("<", "\\u003c")
    metadata = result.get("metadata", {})
    title = html.escape(str(metadata.get("artifact_label", "Cartpole · physics-to-policy gradients")))
    notes = result.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    notes_html = "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)
    raw = html.escape(json.dumps(_clean(result), indent=2, allow_nan=False, ensure_ascii=False))
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:light;--ink:#153148;--muted:#65768a;--line:#dce5ed;--green:#087e74;--blue:#426eba}
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1260px;margin:auto;padding:40px 28px}h1{font-size:38px;line-height:1.15;letter-spacing:-1px;
margin:10px 0 16px}h2{font-size:20px;margin:0 0 14px}h3{font-size:16px;margin:20px 0 10px}
p{margin:9px 0 16px}.eyebrow{color:var(--green);font-size:12px;font-weight:650;letter-spacing:1.7px}
.muted,.small{color:var(--muted)}.small{font-size:12px}.badge{display:inline-block;border-radius:20px;
padding:5px 11px;background:#e5edf5;font-size:12px;margin:4px 6px 0 0}
.notice{padding:14px 18px;border-left:4px solid var(--green);border-radius:8px;background:#e9f4f1;margin:20px 0}
.panel{border:1px solid var(--line);border-radius:14px;padding:23px;background:#fff;margin-top:22px}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:22px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:17px;min-width:0}
.card-title{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.5px}
.value{font-size:26px;font-weight:700;margin:8px 0;line-height:1.2}.good{color:var(--green)}.bad{color:#a1573d}
.controls{display:flex;align-items:center;gap:15px;flex-wrap:wrap;font-size:13px;color:var(--muted)}
select,button{font:inherit;color:var(--ink);background:#fff;padding:7px 10px;border:1px solid var(--line);
border-radius:7px;max-width:100%}select{margin-left:6px}button,summary{cursor:pointer}
.chart{display:block;width:100%;height:auto;margin-top:14px}.chart text{font-size:12px;fill:var(--muted)}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin-top:12px}.scroll{overflow:auto}
table{border-collapse:collapse;width:100%;text-align:left;font-variant-numeric:tabular-nums;font-size:13px;
white-space:nowrap}th{font-size:11px;text-transform:uppercase;letter-spacing:.4px;color:var(--muted);
border-bottom:2px solid var(--line);padding:10px}td{padding:11px 10px;border-bottom:1px solid #edf1f5}
tr:last-child td{border-bottom:0}.workflow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:18px 0}
.workflow div{padding:12px;border:1px solid var(--line);border-radius:9px;font-size:13px;background:#f5f8fa}
.workflow .gradient{border:2px solid var(--green);background:#eaf6f2}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.grid .panel{min-width:0}
details{padding:14px 0;border-top:1px solid var(--line)}summary{font-weight:600}
pre{background:#f3f6f9;border-radius:8px;padding:16px;font:12px/1.6 ui-monospace,monospace;
white-space:pre-wrap;overflow-wrap:anywhere;overflow:auto}.notes{padding-left:20px}.notes li{margin:6px 0}
.tooltip{position:fixed;display:none;pointer-events:none;background:var(--ink);color:white;border-radius:8px;
padding:12px 15px;max-width:320px;white-space:pre-line;font-size:12px;z-index:10}
a{color:var(--blue)}footer{font-size:12px;color:var(--muted);margin-top:24px}
@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}}
@media(max-width:500px){main{padding:24px 14px}h1{font-size:30px}.panel{padding:18px}.value{font-size:23px}}
</style></head><body><main>
<header><div class="eyebrow">ISAAC LAB × ROBOLEARN × EXPERIMENTAL MJWARP ADJOINT</div>
<h1>__TITLE__</h1><p class="muted">Native Isaac Lab Cartpole scene · Newton / MuJoCo Warp physics
· full manager evaluations with the native five-second time limit</p>
<span class="badge" id="budget"></span><span class="badge" id="capture-badge"></span>
<span class="badge">Offline measurements</span>
<p class="notice" id="outcome"></p>
<p class="small">The local GPU is shared with an unrelated workload. Recorded timings describe this experiment;
they are not isolated throughput measurements or a matched PPO speed comparison.</p></header>
<div class="cards" id="proof-cards"></div>
<section class="panel"><h2>Native evaluation quality</h2>
<div class="controls"><label>Metric<select id="metric">
<option value="return_mean">Episode return ↑</option><option value="wrapped_angle_rms">Wrapped angle RMS ↓</option>
<option value="upright_fraction">Upright fraction ↑</option><option value="survival_rate">Timeout survival ↑</option>
<option value="episode_length_mean">Episode length</option>
<option value="action_saturation_fraction">Action saturation</option></select></label>
<label>Horizontal axis<select id="axis"><option value="iteration">Training iterations</option>
<option value="transitions">Training transitions</option></select></label>
<label>Evaluation reset seed<select id="seed"><option value="all">All recorded seeds</option></select></label>
<button id="download" type="button">Download recorded data</button></div>
<svg id="learning" class="chart" viewBox="0 0 1000 400" aria-label="Native evaluation metrics over training"></svg>
<div class="legend" id="legend"></div>
<p class="small">Each dot is a complete native-manager episode evaluation at a recorded checkpoint.
Lines connect measurements with the same evaluation reset seed; they do not supply intermediate evaluations.
The independent final reset seed is shown separately. The zero-action line is a single fixed-seed baseline,
not a training run. Hover a dot for its actual budget and native metrics.</p>
<div class="cards" id="quality-cards"></div>
<p class="small" id="quality-note"></p>
<h3>All recorded native checkpoints</h3><div class="scroll"><table><thead><tr>
<th>Policy</th><th>Iteration</th><th>Reset seed</th><th>Episodes</th><th>Return ± SD</th>
<th>Wrapped RMS (rad)</th><th>Upright</th><th>Survival</th><th>Episode steps</th><th>Saturation</th>
</tr></thead><tbody id="evaluations"></tbody></table></div>
<p class="small">Upright means |wrapped pole angle| &lt; 0.2 rad, averaged over each episode's steps.
Wrapped angle RMS is the square root of the mean per-episode mean squared wrapped angle;
episodes are weighted equally rather than pooling all time steps.
Survival counts timeout without a true failure termination. The native Cartpole task does not terminate merely because
the pole falls; survival alone does not establish balancing.</p>
</section>
<section class="panel"><h2>Finite differences through native physics and the neural actor</h2>
<p>The action probe uniformly perturbs the action across all worlds and differentiates the mean
one-control-step native reward from an identical restored state. Its analytic directional derivative sums
the action gradients across worlds; it does not check every entry of an action Jacobian.
The neural probe differentiates the short-horizon actor objective with respect to one final-layer actor bias.
The critic is initially zero for that probe, so this initial gradient is not supplied by a learned value estimate.</p>
<div class="scroll"><table><thead><tr><th>Probe</th><th>ε</th><th>Analytic gradient</th>
<th>Centered finite difference</th><th>Relative error</th></tr></thead><tbody id="gradients"></tbody></table></div>
<p class="small" id="gradient-rule"></p>
<p class="small">Relative error = |analytic − finite difference| / max(|analytic|, |finite difference|, 10⁻⁸).
All recorded epsilons are shown. These local probes do not validate every network parameter
or every simulator state.</p>
<h3>Native forward parity</h3><div class="scroll"><table><tbody id="native-parity"></tbody></table></div>
<p class="small">This separate recorded probe compares adapter and native-manager observations and rewards.
Forward agreement and finite differences establish different properties.</p>
</section>
<div class="grid"><section class="panel"><h2>Neural parameter changes</h2>
<div class="scroll"><table><tbody id="weights"></tbody></table></div>
<p class="small">Nonzero gradients and parameter changes establish the recorded learning path.
Native policy evaluations determine whether those updates improved control.</p></section>
<section class="panel"><h2>Capture parity</h2><p id="capture-status"></p>
<div class="scroll"><table><tbody id="parity"></tbody></table></div>
<p class="small" id="capture-fields"></p>
<details><summary>Per-array capture errors</summary><pre id="capture-raw"></pre></details>
<p class="small">The harness compares one complete eager update with one graph replay from restored parameters,
optimizer state, and native physics state. Its acceptance limit is 2 × 10⁻⁵ maximum absolute error.
Capture parity does not establish long-run trajectory equivalence.</p></section></div>
<section class="panel"><h2>Recorded budget and timing</h2>
<div class="scroll"><table><tbody id="timing"></tbody></table></div>
<p class="small">The recorded process interval starts inside the harness before scene construction and ends after
the final evaluation, before environment cleanup. The training-and-evaluation interval covers the learning loop
and its checkpoint evaluations; the independent final-seed evaluation follows it.
Per-update timers synchronize the GPU before and after the update. These scopes overlap and are not added together.</p>
<p class="small">The transition budget is the nominal environments × rollout horizon × iterations,
including masked slots after a true termination. Evaluation reset seeds are not independent training seeds.</p>
<div class="controls"><label>Training diagnostic<select id="diagnostic"><option value="actor_loss">Actor loss</option>
<option value="critic_loss">Critic loss</option><option value="seconds">Synchronized update duration (s)</option>
</select></label></div><svg id="training" class="chart" viewBox="0 0 1000 300"
aria-label="Recorded per-update diagnostics"></svg></section>
<section class="panel"><h2>What this experiment establishes</h2>
<div class="workflow"><div>Native Isaac Lab<br>Cartpole scene</div><span>→</span>
<div class="gradient">Neural actor → effort force → native MJWarp adjoint<br>
Short-horizon rewards + frozen critic input gradient</div>
<span>→</span><div>Actor update + detached<br>n-step critic update</div></div>
<p>This is a SHAC-inspired deterministic n-step prototype, not a full SHAC reproduction. It omits SHAC's
stochastic policy, TD-lambda critic, normalization and target-network schedule.
The adapter advances the native Newton/MJWarp Cartpole model using a pinned, unmerged experimental adjoint branch.
The force-input adjoint extension must pass the recorded finite-difference probes before training is interpreted.</p>
<p>Graph capture executes already defined derivatives; capture itself does not make physics differentiable.
This report concerns Cartpole only. It makes no claim about gradients through G1 walking physics or generic
Isaac Lab tasks. Reset operations occur outside the differentiable rollout.</p>
<ul class="notes">__NOTES__</ul><details><summary>Recorded configuration and provenance</summary>
<pre id="metadata"></pre></details>
<details><summary>Complete input result</summary><pre>__RAW__</pre></details></section>
<footer>SHAC: Jie Xu, Viktor Makoviychuk, Yashraj Narang, Fabio Ramos, Wojciech Matusik, Animesh Garg and
Miles Macklin, <a href="https://arxiv.org/abs/2204.07137">Accelerated Policy Learning with Parallel Differentiable
Simulation (2022)</a>. Implementations: <a href="https://github.com/maxkra15/RoboLearn">RoboLearn</a>,
<a href="https://github.com/isaac-sim/IsaacLab">Isaac Lab</a>,
<a href="https://github.com/google-deepmind/mujoco_warp/pull/1535">experimental MuJoCo Warp adjoint</a>,
and NVIDIA <a href="https://nvidia.github.io/warp-nn/">Warp-NN</a>.
All charts and data are embedded; viewing needs no network access.</footer>
</main><div class="tooltip" id="tooltip"></div><script id="data" type="application/json">__DATA__</script><script>
const payload=JSON.parse(document.getElementById('data').textContent),result=payload.result,rows=payload.metrics;
const meta=result.metadata||{},config=result.config||{},evaluations=result.evaluations||[],
baseline=result.zero_action_evaluation;
const $=id=>document.getElementById(id),finite=x=>typeof x==='number'&&Number.isFinite(x);
const fmt=(v,d=3)=>finite(v)?v.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
const compact=v=>Math.abs(v)>=1e6?(v/1e6).toFixed(1)+'M':Math.abs(v)>=1e3?(v/1e3).toFixed(1)+'k':fmt(v,2);
const ns='http://www.w3.org/2000/svg',colors=['#087e74','#426eba','#9b66ae'];
const specs={return_mean:{label:'Episode return',scale:1},wrapped_angle_rms:{label:'Wrapped angle RMS (rad)',scale:1},
upright_fraction:{label:'Upright fraction (%)',scale:100},survival_rate:{label:'Timeout survival (%)',scale:100},
episode_length_mean:{label:'Episode length (steps)',scale:1},
action_saturation_fraction:{label:'Action saturation (%)',scale:100}};
const seeds=[...new Set(evaluations.map(p=>p.seed))].sort((a,b)=>a-b);
const transitions=p=>finite(p.transitions)?p.transitions:
finite(p.iteration)&&finite(meta.num_envs)&&finite(meta.horizon)?
p.iteration*meta.num_envs*meta.horizon:null;
function append(parent,tag,text){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;
parent.appendChild(node);return node;}
function svg(parent,tag,attrs,text){const node=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>
node.setAttribute(k,v));if(text!==undefined)node.textContent=text;parent.appendChild(node);return node;}
function card(parent,title,value,note,good=null){const node=append(parent,'div');node.className='card';
const heading=append(node,'div',title);heading.className='card-title';const number=append(node,'div',value);
number.className='value'+(good===true?' good':good===false?' bad':'');append(node,'div',note).className='small';}
function tableRow(parent,label,value){const row=append(parent,'tr');append(row,'td',label);append(row,'td',value);}
function tooltip(event,text){const box=$('tooltip');box.textContent=text;box.style.display='block';
box.style.left=Math.max(10,Math.min(event.clientX+12,innerWidth-box.offsetWidth-10))+'px';
box.style.top=Math.max(10,Math.min(event.clientY+12,innerHeight-box.offsetHeight-10))+'px';}
function frame(canvas,points,height,percent=false){const low=Math.min(0,...points.map(p=>p.y)),
high=Math.max(percent?100:1,
...points.map(p=>p.y)),pad=percent?0:Math.max(.01,(high-low)*.06),ymin=percent?0:low-pad,ymax=percent?100:high+pad;
const xmax=Math.max(1,...points.map(p=>p.x)),bottom=height-55;
const x=v=>85+v/xmax*855,y=v=>bottom-(v-ymin)/(ymax-ymin)*(height-105);
for(let i=0;i<=5;i++){const a=xmax*i/5,b=ymin+(ymax-ymin)*i/5;
svg(canvas,'line',{x1:85,y1:y(b),x2:940,y2:y(b),stroke:'#e5edf3'});
svg(canvas,'text',{x:75,y:y(b)+4,'text-anchor':'end'},compact(b));
svg(canvas,'text',{x:x(a),y:height-28,'text-anchor':'middle'},compact(a));}return {x,y};}
function redraw(){const canvas=$('learning');canvas.replaceChildren();$('legend').replaceChildren();
const key=$('metric').value,spec=specs[key],axis=$('axis').value,seed=$('seed').value;
const shown=evaluations.filter(p=>(seed==='all'||String(p.seed)===seed)&&finite(p[key]));
const points=shown.map(p=>({p,x:axis==='iteration'?p.iteration:transitions(p),y:p[key]*spec.scale}))
.filter(p=>finite(p.x));
const zero=baseline&&finite(baseline[key])?baseline[key]*spec.scale:null;
if(!points.length){svg(canvas,'text',{x:30,y:85},'No native evaluation measurements recorded');return;}
const axes=frame(canvas,zero===null?points:[...points,{x:0,y:zero}],400,spec.scale===100);
svg(canvas,'text',{x:85,y:24},spec.label);svg(canvas,'text',{x:510,y:393,'text-anchor':'middle'},
axis==='iteration'?'Completed training iterations':'Training transitions (environments × horizon × iterations)');
seeds.forEach((value,index)=>{const line=points.filter(p=>p.p.seed===value).sort((a,b)=>a.x-b.x);if(!line.length)return;
const legend=append($('legend'),'span','Evaluation reset seed '+value);legend.style.color=colors[index%colors.length];
svg(canvas,'polyline',{points:line.map(p=>`${axes.x(p.x)},${axes.y(p.y)}`).join(' '),fill:'none',
stroke:colors[index%colors.length],'stroke-width':2.4});line.forEach(({p,x,y})=>{
const dot=svg(canvas,'circle',{cx:axes.x(x),cy:axes.y(y),r:5.5,fill:colors[index%colors.length],
stroke:'white','stroke-width':1.4});
const text=`Iteration ${fmt(p.iteration,0)} · reset seed ${p.seed}\n${fmt(transitions(p),0)} training transitions\n`+
`Return ${fmt(p.return_mean)} ± ${fmt(p.return_std)} episode SD\nWrapped angle RMS ${fmt(p.wrapped_angle_rms)} rad\n`+
`Upright ${fmt(p.upright_fraction*100,1)}% · timeout survival ${fmt(p.survival_rate*100,1)}%\n`+
`${fmt(p.episodes,0)} native episodes`;
dot.onpointermove=event=>tooltip(event,text);dot.onpointerleave=()=>{$('tooltip').style.display='none';};
svg(dot,'title',{},text);});});
if(zero!==null){svg(canvas,'line',{x1:85,y1:axes.y(zero),x2:940,y2:axes.y(zero),stroke:'#b07845',
'stroke-dasharray':'6 4'});
append($('legend'),'span','Zero actions · reset seed '+baseline.seed).style.color='#b07845';}}
function diagnostics(){const canvas=$('training');canvas.replaceChildren();const key=$('diagnostic').value;
const points=rows.filter(p=>finite(p.iteration)&&finite(p[key])).map(p=>({x:p.iteration,y:p[key]}));
if(!points.length){svg(canvas,'text',{x:30,y:80},'No per-update diagnostic rows available');return;}
const axes=frame(canvas,points,300);svg(canvas,'text',{x:85,y:24},$('diagnostic').selectedOptions[0].textContent);
svg(canvas,'polyline',{points:points.map(p=>`${axes.x(p.x)},${axes.y(p.y)}`).join(' '),fill:'none',
stroke:'#426eba','stroke-width':1.6});
svg(canvas,'text',{x:510,y:293,'text-anchor':'middle'},'Completed training iterations');}
seeds.forEach(value=>{const option=append($('seed'),'option','Seed '+value);option.value=String(value);});
['metric','axis','seed'].forEach(id=>$(id).onchange=redraw);$('diagnostic').onchange=diagnostics;
const gradients=result.gradients||{},proof=result.weight_proof||{},capture=result.capture_parity;
card($('proof-cards'),'Gradient probes',gradients.accepted===true?'Accepted':
gradients.accepted===false?'Rejected':'Not recorded',
'One action probe + one neural bias probe',gradients.accepted);
card($('proof-cards'),'Actor gradient norm',fmt(gradients.actor_gradient_norm,6),'Initial objective gradient L2 norm');
card($('proof-cards'),'Actor / critic updates',fmt(proof.actor_updates,0)+' / '+fmt(proof.critic_updates,0),
'Recorded optimizer timesteps');
card($('proof-cards'),'Training transitions',fmt(result.transitions,0),
fmt(result.iterations,0)+' completed iterations');
$('budget').textContent=fmt(meta.num_envs,0)+' environments × '+fmt(meta.horizon,0)+' rollout steps';
$('capture-badge').textContent=result.capture===true?'Captured update':
result.capture===false?'Eager update':'Capture status not recorded';
$('outcome').textContent=meta.outcome_summary||
(gradients.accepted===false?'Finite-difference gate rejected: these diagnostics do not validate policy learning.':
proof.finite!==undefined?'Recorded outcome: inspect native return, angular control and timeout survival together.':
'Gradient diagnostics or unfinished training: native learning outcomes are shown only where recorded.');
const maximum=Math.max(-1,...evaluations.map(p=>p.iteration).filter(finite));
const latest=evaluations.filter(p=>p.iteration===maximum).sort((a,b)=>a.seed-b.seed)[0];
const independent=evaluations.find(p=>p.iteration===maximum&&p.seed!==latest?.seed);
if(!meta.outcome_summary&&gradients.accepted!==false&&latest){
const final=independent||latest,primary=evaluations.filter(p=>p.seed===latest.seed&&finite(p.return_mean));
const best=primary.reduce((a,b)=>a.return_mean>b.return_mean?a:b);
$('outcome').textContent='Final policy at iteration '+fmt(maximum,0)+' · evaluation reset seed '+final.seed+
': native return '+fmt(final.return_mean)+', upright '+fmt(final.upright_fraction*100,1)+
'%, timeout survival '+fmt(final.survival_rate*100,1)+'%. '+
'Primary reset seed '+latest.seed+' had its highest recorded return '+fmt(best.return_mean)+
' at iteration '+fmt(best.iteration,0)+'; its final return was '+fmt(latest.return_mean)+
'. This is one training seed, with distinct evaluation reset seeds.';}
for(const key of ['return_mean','wrapped_angle_rms','upright_fraction','survival_rate']){
const spec=specs[key],value=latest&&finite(latest[key])?latest[key]*spec.scale:null;
const sameSeed=latest&&baseline&&latest.seed===baseline.seed&&finite(baseline[key]);
const delta=sameSeed&&finite(value)?value-baseline[key]*spec.scale:null;
card($('quality-cards'),spec.label,fmt(value,key==='return_mean'?3:spec.scale===100?1:3)+(spec.scale===100?'%':''),
delta===null?'No matching zero-action baseline':
(delta>0?'+':'')+fmt(delta,3)+' versus zero actions (same reset seed)');}
$('quality-note').textContent=latest?'Cards use iteration '+latest.iteration+
' and primary evaluation reset seed '+latest.seed+
'. Other recorded evaluation seeds remain separate in the plot and table.':'No native evaluation outcomes recorded.';
for(const point of [...(baseline?[{...baseline,iteration:0,zero:true}]:[]),...evaluations]){
const row=append($('evaluations'),'tr');[point.zero?'Zero actions':'Actor policy',fmt(point.iteration,0),
String(point.seed??'—'),
fmt(point.episodes,0),fmt(point.return_mean)+' ± '+fmt(point.return_std),fmt(point.wrapped_angle_rms),
finite(point.upright_fraction)?fmt(point.upright_fraction*100,1)+'%':'—',
finite(point.survival_rate)?fmt(point.survival_rate*100,1)+'%':'—',fmt(point.episode_length_mean,1),
finite(point.action_saturation_fraction)?fmt(point.action_saturation_fraction*100,1)+'%':'—']
.forEach(v=>append(row,'td',v));}
if(!$('evaluations').children.length)tableRow($('evaluations'),'No native evaluations recorded','—');
for(const [key,label] of [['physics_action','Uniform action → mean one-step reward'],
['actor_parameter','Actor bias → short-horizon objective']]){
for(const probe of gradients[key]||[]){const row=append($('gradients'),'tr');[label,fmt(probe.epsilon,5),
fmt(probe.analytic,8),fmt(probe.finite_difference,8),
finite(probe.relative_error)?fmt(probe.relative_error*100,2)+'%':'—'].forEach(v=>append(row,'td',v));
row.lastChild.className=finite(probe.relative_error)&&probe.relative_error<.05?'good':'bad';}}
if(!$('gradients').children.length)tableRow($('gradients'),'No finite differences recorded','—');
$('gradient-rule').textContent='Harness acceptance: at least one recorded ε has relative error < 5% for each probe. '+
'Result: '+
(gradients.accepted===true?'accepted.':gradients.accepted===false?'rejected.':'not recorded.');
const native=result.native_parity;
if(native){tableRow($('native-parity'),'Harness forward-parity acceptance',
native.accepted===true?'Accepted':native.accepted===false?'Rejected':'Not recorded');
tableRow($('native-parity'),'Observation maximum absolute error',fmt(native.observation_max_abs_error,9));
tableRow($('native-parity'),'Reward maximum absolute error',fmt(native.reward_max_abs_error,12));}
else tableRow($('native-parity'),'Native forward probe','Not recorded');
for(const [label,key] of [['Finite parameters and Adam state','finite'],
['Actor parameter Δ L2','actor_parameter_delta_l2'],
['Critic parameter Δ L2','critic_parameter_delta_l2'],['Actor optimizer timesteps','actor_updates'],
['Critic optimizer timesteps','critic_updates']])
tableRow($('weights'),label,key==='finite'?proof[key]===true?'Yes':proof[key]===false?'No':'—':
fmt(proof[key],key.endsWith('updates')?0:7));
$('capture-status').textContent=capture?'Recorded eager-versus-replay comparison.':
result.capture===false?'Eager run; capture parity was not requested.':'No capture parity recorded.';
if(capture){tableRow($('parity'),'Native qpos maximum absolute error',fmt(capture.qpos_max_abs_error,9));
if('qvel_max_abs_error' in capture)
tableRow($('parity'),'Native qvel maximum absolute error',fmt(capture.qvel_max_abs_error,9));
for(const [label,key] of [['Parameter arrays','parameter_max_abs_errors'],
['Optimizer arrays','optimizer_max_abs_errors']]){
const errors=Object.values(capture[key]||{}).filter(finite);if(!errors.length)continue;
tableRow($('parity'),label+' · maximum across '+errors.length+' arrays',Math.max(...errors).toExponential(3));}
$('capture-raw').textContent=JSON.stringify(capture,null,2);
$('capture-fields').textContent='Only the recorded array groups summarized above are numerically compared. '+
'Restoring optimizer state is distinct from comparing its numerical results.';}
else{tableRow($('parity'),'Comparison','Not recorded');$('capture-raw').parentElement.remove();}
const updateSeconds=rows.map(p=>p.seconds).filter(finite);
for(const [label,value] of [['Completed iterations',fmt(result.iterations,0)],
['Training transitions',fmt(result.transitions,0)],
['Control dt',fmt(meta.control_dt,8)+' s'],['Physics dt',fmt(meta.physics_dt,8)+' s'],
['Training + checkpoint-evaluation interval',fmt(result.training_and_evaluation_seconds,2)+' s'],
['Recorded process interval',fmt(result.process_seconds,2)+' s'],
['Recorded synchronized updates',fmt(updateSeconds.length,0)],
['Sum of measured update durations',updateSeconds.length?fmt(updateSeconds.reduce((a,b)=>a+b,0),2)+' s':'—']])
tableRow($('timing'),label,value);
$('metadata').textContent=JSON.stringify({metadata:meta,config,notes:result.notes},null,2);
$('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],
{type:'application/json'}));
const link=document.createElement('a');link.href=url;link.download='differentiable-cartpole-data.json';link.click();
setTimeout(()=>URL.revokeObjectURL(url),1000);};redraw();diagnostics();
</script></body></html>"""
    for key, value in (("TITLE", title), ("DATA", encoded), ("NOTES", notes_html), ("RAW", raw)):
        template = template.replace(f"__{key}__", value)
    return template


def main() -> None:
    """Load the harness result and optional adjacent per-update metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Recorded results.json from the harness.")
    parser.add_argument("--output", type=Path, required=True, help="Destination offline HTML file.")
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    metrics_path = args.input.with_name("metrics.jsonl")
    metrics = []
    if metrics_path.is_file():
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(result, metrics))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
