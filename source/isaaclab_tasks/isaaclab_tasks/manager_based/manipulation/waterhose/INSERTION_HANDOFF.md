# Waterhose Cable‑Insertion — Engineering Handoff

## 0. Mandate (read first)

The scripted demo `run_robot_demo.py` must end‑to‑end **grasp the plug on the flexible
cable and insert it into the socket** inside Newton's coupled MuJoCo + VBD solver, without
the cable going unstable.

Today the cable is **stable** and the **grasp works**, but the cable's tail is held by an
artificial Newton *fixed joint* to a kinematic point ("Anchor1"). That weld is the part we
distrust and the part that historically blew up on contact. The hypothesis below (a **fixed
SDF "plug/anchor"** that mechanically captures the cable end, like the reference does) is the
lead idea, but **you own the solution**.

**Be very explorative.** Try the SDF‑anchor idea, try the reference‑faithful rebuild, try
grip/contact tuning — whatever it takes. **Do not stop** until either (a) `run_robot_demo.py`
inserts the plug end‑to‑end repeatably, or (b) you have *conclusive, evidenced* root‑cause for
why it cannot, with the data to back it up. Keep a short log of what you tried and what the
sim did (the debug prints below make this easy).

---

## 1. Background / what this demo is

- An RBY1 dual‑arm robot (28‑DOF, fixed base) must pick a **plug** that sits at the head of a
  **flexible cable (1D rod)** and **insert** it into a **socket**.
- Physics is **Newton's coupled solver**: a **MuJoCo** sub‑solver owns the rigid robot, a
  **VBD** sub‑solver owns the deformable cable + the plug. The two are linked by **proxy
  coupling**: selected robot gripper bodies are mirrored into the VBD view as "proxies" so the
  cable can collide with the fingers.
- A **scripted state machine** (not a learned policy) drives the right arm through phases:
  `REST → APPROACH → ENGAGE → GRASP → HOLD_GRASP → RETRACT → … → ALIGN_AXES → VERIFY_ALIGN →
  INSERT → DONE`. A force‑feedback loop closes the gripper until a target grip force is read
  back from the proxy coupling wrenches.

### Current status
- ✅ Cable is numerically **stable** for the whole run (no explosion when the gripper arrives).
- ✅ Gripper **grasps and holds the plug**; the plug + cable head are dragged with the hand.
- ⚠️ **Insertion not yet verified** end‑to‑end.
- ⚠️ **Grip force massively overshoots** (target 80 N, measured ~6–7 kN) — the loop slams to
  full close before the fingers reach the plug, then the very stiff finger actuator crushes it.
- ⚠️ **Tail anchor is a fixed‑joint weld to a kinematic sphere** — the artifact we want to remove.

---

## 2. What seems to be the issue

The **cable endpoint rigging** is the weak point.

We currently weld the cable **tail** (last rod segment) to a 1 mm **kinematic** sphere
(`Anchor1`) via a Newton `add_joint_fixed`, and weld the **head** (segment 0) to the `Plug1`
rigid body. The head weld is fine/needed (it's the graspable plug). The **tail weld is the
problem child**:

- It is an **artificial pin**, not how the cable would really be held.
- It is a **stiff constraint** that fights contacts. The historical "touch → explode" came from
  this stiff pin + light gripper proxies (`mass_scale=1`) pumping energy: a free cable would
  absorb a gripper shove by translating, but a tail‑pinned soft cable reflects the impulse and
  detonates. (We tamed it by making proxies immovable — see §4 — but the pin is still unphysical
  and brittle to tune.)
- The **reference does not use any weld** (see §3). Its cable is free and held purely by grip +
  a static **SDF connector** the cable end rests in.

So: the instability was *mostly* `mass_scale`, now fixed — but the tail weld remains the thing
that makes the setup fragile and "weird," and is the most likely blocker to clean insertion.

---

## 3. The idea to implement (lead hypothesis)

**Replace the tail fixed‑joint weld with a fixed SDF "plug/anchor."**

Author a **static collider** (a small connector/socket body) with an **SDF mesh collider**,
placed at the cable's tail. Shape it so the cable end **threads into it and is mechanically
captured** — it physically cannot slide out — so the cable is held by **contact**, not by a
rigid joint. This is compliant and robust (contacts absorb impulses) and removes the artificial
constraint entirely.

This is **exactly how the Newton reference holds its cable**: it loads an STL hose‑connector,
builds an SDF for it, and places it as static scene geometry; the cable's straight end sits in
that connector and is held by contact + the robots' grip. No weld.

**Concrete SDF mechanism in Newton** (already used by the reference):
```python
mesh = newton.Mesh(vertices, indices, compute_inertia=True, is_solid=True)
if wp.get_device().is_cuda:
    mesh.build_sdf(max_resolution=64)   # <-- builds the SDF collider
```
Reference: `example_cable_robot_proxy_coupled_solver.py` ~lines 1238–1252 (`_load_stl_as_tri_mesh`,
`newton.Mesh(...)`, `mesh.build_sdf(...)`, then positioned at the hose layout).

### Things to work out while implementing
- **How the cable end engages the SDF.** Geometry/clearance so the rod end is captured but not
  crushed; cable radius vs. socket bore; contact `mu`, `margin`, `gap` (reference uses
  `mu=1.0, margin=0.0, gap=0.001`).
- **Where the SDF body lives.** It must be in the **VBD** view to collide with the cable
  (the cable lives in VBD). See the body_entities / `include_static_shapes` notes in §5.
- **Whether to keep the head plug weld** (likely yes — it is the graspable body) or also make the
  head a free segment captured by the gripper like the reference (the reference welds nothing).
- **Whether to keep `Anchor1` at all** once the SDF holds the tail. Ideally delete it and remove
  its `CableAttachmentCfg`.

> Note: our cable‑attachment code *already* supports welding to a **static USD prim** (not a
> body) — see `_resolve_static_target_xform` in `cable_object.py`. That's a weld, not an SDF
> capture, but it shows the plumbing for "attach cable to a static scene element," which you may
> reuse or replace.

---

## 4. The current stable baseline (so you don't regress it)

Solver / coupling (in `waterhose_env_cfg.py`):
- Coupling: `proxy`; finger bodies only are proxies; `mass_scale=1.0e3` (immovable),
  `collide_interval=1`, `mode="lagged"`, proxy pipeline `contact_matching="sticky"`.
- `num_substeps=10`, VBD `iterations=10`, `rigid_contact_hard=False`, `rigid_joint_hard=False`
  (VBD structural joints are softened to penalty‑only via a post‑build hook — see coupled_manager).
- `model_cfg`: `shape_material_ke=1e5`, `kd=1e-1`, `soft_contact_mu=1.0`, `shape_material_mu=1.0`.

Cable1 material:
- `stretch_stiffness (EA)=1e6`, `bend_stiffness=3.0`, `stretch_damping=1e-3`, `bend_damping=1e0`,
  `density=1000`, `resample_segment_length=None`.

Attachments (the thing under review):
- Head: weld segment `0` → `Plug1`, `cable_local_pos=(0,0,0.022)`.
- Tail: weld segment `-1` → `Anchor1` (kinematic sphere). **← replace with SDF capture.**

Gripper:
- Actuators: gripper drive `stiffness=10000`, finger joints `stiffness=500000` (very stiff —
  this is why grip force overshoots).
- Scripted grip loop: target `80 N`, tighten rate `0.8`, max‑close command `-1.0`.

> ⚠️ The key lever that fixed the contact explosion was **`mass_scale=1.0e3` + `collide_interval=1`**
> on the proxy. If you rebuild the rigging, keep proxies effectively immovable or the
> "touch → explode" returns.

---

## 5. Where everything lives (file map)

### The gold‑standard reference (study this first)
`/home/maximiliank/Work/newton/newton/examples/cable_robot/example_cable_robot_proxy_coupled_solver.py`
- `_create_cable_objects` (~1040): builds rods with `builder.add_rod(...)`, **no weld**; finds the
  grasp segment nearest the capsule.
- `_compute_capsule_specs` + cable build entry (~1231): capsule "socket" geometry the cable rests in.
- **SDF connector mesh** (~1238–1252): STL → `newton.Mesh(...)` → `mesh.build_sdf(max_resolution=64)`.
- Cable material (~600–604): `EA=1e12`, `density=10000`, `bend_rigidity=3.0`, `stretch_damping=1e-3`,
  `bend_damping=1e0`.
- Proxy bodies = gripper **base + both fingers** (~1178–1185); `contact_matching="sticky"` (~1416).
- Extract/inject state machine (~197–310, `extract_distance` ~1614) — analogous to our insert.
- NOTE: this file needed an import patch to run: `from newton.solvers.experimental.coupled import
  ModelView, SolverCoupled, SolverCoupledProxy as SolverProxyCoupled`.

### Our task (the thing to make work)
Base dir: `/home/maximiliank/Work/IsaacLab-waterhose-demo/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/waterhose/`
- `waterhose_env_cfg.py` — scene + solver config. Key spots: `anchor1` (~237), `plug1` + `cable1` +
  `attachments` (~285–322), proxy coupling + `mass_scale` (~497–520), `model_cfg`/`num_substeps` (~520+).
- `scripted_state_machine.py` — phases enum (~64–93), **socket pose** `socket_pos_w` +
  `socket_quat_w` (~166–173), grip loop `_update_grip_command` (~438), grip‑force reader
  `_get_right_proxy_grip_force` (~460), live plug/cable frame `_get_live_plug_frame` (~484).
- `mdp/actions.py` — `WaterhoseGripperPositionAction`: maps one scalar to finger targets
  (`open_command_expr` / `close_command_expr`); `close_alpha = (1-action)*0.5`.
- `assets/fridge/cable/plug.usda` — plug geometry (mass set to 0.001).
- `assets/fridge/cable/cable001.usda` — the cable polyline asset.
- `assets/fridge/…` — fridge USD (~73 MB collision mesh; kept out of VBD via `include_static_shapes=False`).

Run script:
- `/home/maximiliank/Work/IsaacLab-waterhose-demo/scripts/environments/waterhose/run_robot_demo.py`
  — entry point; contains the temporary `_debug_cable_positions` helper and `--vis/--max_steps/
  --settle_time/--debug_script` flags. (Temporary debug + unused `_ANCHOR2/_CABLE2` constants
  should be cleaned up once it works.)

### Framework plumbing (where the cable/coupling is actually built)
- `…/isaaclab_contrib/isaaclab_contrib/cable/cable_object.py`
  - `add_rod` build, `apply_cable_attachments_to_builder` (~224) → **`add_joint_fixed` (~306)**
    ← this is the weld to remove/replace; `_resolve_static_target_xform` (~202) ← weld‑to‑static‑prim
    plumbing; arc‑length resampling helper.
- `…/isaaclab_contrib/isaaclab_contrib/cable/cable_object_cfg.py` — `CableObjectCfg`,
  `CableAttachmentCfg`, `resample_segment_length`.
- `…/isaaclab_newton/isaaclab_newton/physics/coupled_manager.py` — proxy solver build,
  `_apply_vbd_joint_constraint_modes` (softens VBD joints when `rigid_joint_hard=False`),
  `get_proxy_body_wrenches` (the grip‑force source).
- `…/isaaclab_newton/isaaclab_newton/physics/coupled_manager_cfg.py` — `ProxyCouplingCfg`,
  `CoupledProxyCfg` (`mass_scale`), `CoupledSolverCfg`.
- `…/isaaclab_contrib/isaaclab_contrib/deformable/newton_manager_cfg.py` — `VBDSolverCfg`
  (`rigid_contact_hard`, `rigid_joint_hard`, `rigid_contact_history`, joint k‑starts, …),
  `NewtonModelCfg`.
- `…/isaaclab_newton/isaaclab_newton/physics/newton_collision_cfg.py` and
  `…/isaaclab_newton/isaaclab_newton/sim/schemas/schemas_cfg.py` — collision pipeline / shape
  schema config (where collision‑mesh / SDF approximation options would be wired).

### Secondary reference (open in the IDE; another deformable hose example)
- `/home/maximiliank/Work/IsaacLab-anymal-firehose/scripts/demos/newton_anymal_mpm_firehose.py`
  — ANYmal + MPM firehose; useful for how a long flexible hose is anchored/handled in this stack.

---

## 6. How to run & read the result

```bash
# from /home/maximiliank/Work/IsaacLab-waterhose-demo  (branch: max/waterhose-coupled-experimental)
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
    --task Isaac-Waterhose-Coupled-v0 --vis none \
    --max_steps 480 --settle_time 0.5 --debug_script
```
- `--vis kit` to watch it; `--vis none` for fast headless iteration.
- `--debug_script` prints `[waterhose_ik] PHASE: … grip_force=… finger_gap=… tip_axis_cos=… pos_err=…`
  at each phase change.
- `_debug_cable_positions` prints `[cable-debug] step=N: n=… finite=… min=… max=… first=…`.

**Healthy:** `finite=True` and bounds stay O(0.1 m) throughout; `grip_force>0` at/after GRASP;
`plug_w` follows the hand on RETRACT; INSERT drives the plug toward `socket_pos_w`
(`[-0.2594, 0.3630, 0.2373]`, tilted ~20° about X); state reaches DONE.

**Unhealthy:** `[cable-debug]` values jump to O(1–10+ m) or `finite=False` (explosion);
`grip_force=0` through GRASP (never contacts); plug drifts away from finger midpoint (slip).

---

## 7. Candidate avenues (explore broadly — not exhaustive)

1. **SDF tail anchor (primary).** Static SDF connector capturing the cable end; delete `Anchor1`
   + its attachment. Mirror the reference's `newton.Mesh(...).build_sdf(...)` static geometry.
2. **Reference‑faithful rebuild.** No welds at all; hold cable by SDF connector + grip; move
   `EA`/`density` toward the reference (`1e12`/`1e4`) and re‑verify stability with immovable proxies.
3. **Grip‑force regulation.** Soften finger actuator (`stiffness=500000` is the overshoot cause)
   and/or start the close nearer the plug so the 80 N feedback target actually holds.
4. **Grasp alignment.** `tip_axis_cos ≈ -0.5` at GRASP (should be ≈ -1). Fix grasp orientation so
   the fingers close across the plug axis — improves hold and insertion.
5. **If keeping a weld:** it is currently penalty‑soft (`rigid_joint_hard=False`); compare
   hard vs. soft, joint k‑start ramps, and `mass_scale` sensitivity, and document why.

Whatever path you take, **keep proxies effectively immovable** (`mass_scale` high,
`collide_interval=1`) or the contact explosion returns.
