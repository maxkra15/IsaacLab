# Waterhose grasp-and-insert demo (RBY1DF)

A two-arm RBY1DF robot grasps a water-hose connector and inserts it into a fridge socket, using a
Newton coupled solver: **MuJoCo-Warp (MJWarp)** for the articulated rigid robot and **VBD** for the
deformable hose (a Cosserat rod), joined by two-way coupling. The hose head carries the rigid plug;
its tail is welded to a kinematic anchor. A scripted state machine drives the demo; the same scene is
also teleoperable (SpaceMouse / keyboard / Apple Vision Pro XR).

## Registered tasks

| Task id | Coupling | Action space | Use |
|---|---|---|---|
| `Isaac-Waterhose-Coupled-v0` | proxy | scripted multi-body Newton-IK | **Default customer demo** (single env) |
| `Isaac-Waterhose-Admm-v0` | ADMM | scripted multi-body Newton-IK | Solver comparison; **the batchable path** |
| `Isaac-Waterhose-Coupled-Teleop-v0` | proxy | relative Newton-IK (right EE), torso+left pinned | Teleop (SpaceMouse/keyboard/XR) |

`WaterhoseIkEnvCfg` (base DiffIK) exists as an unregistered reference variant only.

## Running

```bash
# Scripted demo, Newton viewer (kitless — no Omniverse Kit):
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --visualizer newton

# Scripted demo headless with a throughput print:
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py --headless --visualizer none --profile

# Teleop (SpaceMouse / keyboard / XR):
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Waterhose-Coupled-Teleop-v0 <device / --xr flags>
```

The coupled tasks run on a kitless Newton backend. With no `--visualizer` they boot Omniverse Kit
(default rendering); pass an explicit kitless visualizer (`--visualizer newton|rerun|viser|none`) to
skip Kit entirely. See `run_robot_demo.py:_coupled_task_needs_kit` for the exact rule.

### Environment-variable flags
- `WATERHOSE_FRIDGE_BODY_COLLISION` (default **on**): robot↔housing collision — routes the ~245 convex
  housing hulls into the MJWarp (robot) entry. `0` = robot collides with its own shapes only.
- `WATERHOSE_HOSE_BODY_COLLISION` (default **on**): hose↔housing-body collision — routes the single
  welded concave body mesh into the VBD (hose) entry (a <1 mm graze of the hose against the fridge body).
  The mesh is decimated to ~700 triangles (see `simplify_body_collision.py`) so it batches to 128 envs.
  Set `0` for ~1.5× more throughput (drops the graze).
- `WATERHOSE_SOCKET_SDF` (default **off**): the socket bore is a plain triangle-mesh (BVH) collider;
  `1` upgrades it to a texture-SDF (smoother insertion gradient, but a per-env SDF build at startup).
  The SDF is applied in code (`spawn_fridge_with_socket_sdf`), not baked in the USD, so this flag is the
  single source of truth.
- `WATERHOSE_ASSETS_DIR`: override the packaged asset root.

## Parallelization model
The env is cloned the best-practice way: all scene assets live on `/World/envs/env_.*/...` prim paths,
`replicate_physics=True` (correct for the Newton backend, whose replication doesn't use the PhysX
deformable parser). The coupled Newton backend **parses each USD once** into an env_0 source builder,
then replicates it per env via `add_builder` inside per-world contexts — so the heavy meshes (fridge,
robot) are loaded once and the **mesh geometry is shared across all envs** (verified: at N envs each
shape has N instances but a single shared geometry object), which is why memory stays flat. Inter-env
collisions are filtered by Newton's per-world model (the broad phase rejects cross-world pairs), not by
the PhysX `filter_collisions` path. The cable tail anchor is a per-env body (a fixed joint to the shared
world body NaNs the multi-env solve), and all `init_state.pos` values are interpreted per-env. The only
env_0 literal is the teleop `target_frame_prim_path` (`_ROBOT_BASE_PRIM_PATH_ENV0`), confined to the
documented single-env teleop path.

## Performance & batchability (see the scaling report)

Full numbers and plots: `_scratch/reports/waterhose_scaling/waterhose_env_scaling_report.tex`
(data CSV + bench drivers beside it). Measured on an RTX 5090.

- **Single env:** proxy (default) ≈ 26 env-steps/s (38 ms/step); ADMM ≈ 14 env-steps/s. Per step the
  cost is the VBD/AVBD cable+contact solve (42%), the MJWarp robot solve (25%), and mesh collision
  (11%); the CUDA graph is essential (~6× over eager).
- **Batched (all collisions on):** throughput rises monotonically and **peaks at 2048 worlds** (the
  optimum on a 32 GiB RTX 5090); 4096/8192 don't complete (build time + memory). Throughput-bound, not
  memory-bound (thin Cosserat rod). Startup grows to ~7 min at 2048 (1024 ≈ 2.5 min is the iter sweet
  spot).
  - **ADMM** is fastest at the optimum: ≈ **13,053 env-steps/s at 2048 worlds** (14 / 1,490 / 9,343 /
    13,053 at n=1/128/1024/2048).
  - **Proxy** (default) ≈ 11,032 env-steps/s at 2048; it leads ADMM up to ~1024 worlds, ADMM above.
  - `WATERHOSE_HOSE_BODY_COLLISION=0` drops the hose↔body graze for extra throughput.

### The hose-body mesh (decimated so it batches)
The hose↔fridge-body contact routes a welded mesh into the proxy `CollisionPipeline`, whose hose-particle
mesh-triangle candidates feed a **global `triangle_pairs` buffer hard-capped at `2^20`** by deterministic
contact matching (the `contact_matching="latest"` the grip warm-start needs). The mesh originally had
**~103k triangles**, which overflowed that cap past ~4 envs (`CUDA error 700`). It is now **decimated to
~700 triangles** (`scripts/environments/waterhose/simplify_body_collision.py`, a numpy vertex-clustering
pass that rewrites it in place in `fridge.usda`), so the contact batches to 128 envs — hence on by
default. With it on the hose drags against the body, so the scripted carry/align phases take longer
(still reaches DONE); set `WATERHOSE_HOSE_BODY_COLLISION=0` to drop it for max throughput. The
robot↔housing hulls and plug↔socket contact are independent.

The manager-level code (state machine, terminations, actions, per-env Newton-body resolution, and the
model-init hooks) is correctly vectorized; the only single-env-specific code is the env-0 phase
print-out (cosmetic).

## Architecture map

- **`waterhose_env_cfg.py`** — scene, MDP, the `CoupledNewtonCfg` solver (MJWarp + VBD entries, proxy
  and ADMM coupling), the action cfgs, and the env variants. Physics tuning constants live at the top
  (`_VBD_*`, `_GRIPPER_*`). Three `MODEL_INIT` builder hooks run before `finalize()`:
  `_restrict_rby1df_collision_to_right_gripper`, `_disable_anchor_collision`,
  `_merge_plug_shape_into_cable_head` (re-parents the plug shape onto the cable head body so the
  connector is rigidly part of the rod).
- **`scripted_state_machine.py`** — the demo policy (REST→…→DONE). Reads the live plug pose from the
  Newton solver state (so the pick is agnostic to the cable's resting pose), emits the multi-body IK
  action (right EE + left/torso holds) + gripper.
- **`mdp/actions.py`** — `WaterhoseGripperPositionAction`,
  `WaterhoseLocalFrameNewtonInverseKinematicsAction` (relative-EE teleop, EE-frame roll), and
  `WaterhoseTeleopPinnedNewtonIkAction` (teleop with torso+left pinned).
- **`mdp/terminations.py`** — `plug_inserted_in_socket` success predicate.
- **`geometry.py`** — shared poses/offsets/collider label patterns. **All quaternions are `(x, y, z, w)`.**
- **`teleop.py` / `teleop_pipelines.py`** — SpaceMouse device + IsaacTeleop XR pipelines.

## Handover gotchas
- **Quaternions are `(x, y, z, w)` everywhere** (Isaac Lab math + Newton IK). USD-authored `(w,x,y,z)`
  is converted in `geometry.py`.
- **Do not delete `teleop_pipelines_legacy.py`.** It is a deliberate byte-for-byte fallback of the
  last known-good XR pipeline; switch the import in `waterhose_env_cfg.py` back to it if a pipeline
  refactor regresses the live session.
- **The cable tail anchor must be a per-env body**, not the shared world body (`-1`) — a fixed joint to
  the world body NaNs the multi-env coupled solve at step 0.
- The plug's `RigidObject` asset view is stale for a coupled body; read its pose from
  `NewtonManager.get_state_0().body_q` (the SM and the success term already do this).
- The socket/connector contact stiffness comes from the model-wide `NewtonModelCfg.shape_material_*`
  fill, which overwrites per-shape USD/material contact properties — tune it there, not on the USD.
- **Viewer geometry (Visuals vs Collisions).** Kit renders by USD `purpose`: the fridge/robot render
  meshes are `default`/`render` (shown), the colliders are `purpose=guide` (hidden by default; toggle
  via Kit's Guide display-purpose). The **Newton** viewer instead keys on `ShapeFlags`: a shape shows
  under *Visuals* iff `VISIBLE`, under *Collisions* iff `COLLIDE_SHAPES`. The Newton USD importer sets
  `VISIBLE` on the fridge colliders even though they're `guide`, so `_hide_fridge_collider_visuals`
  (a MODEL_INIT callback) clears it — fridge colliders then live only under *Collisions*, leaving the
  fridge visual mesh under *Visuals*. The Newton cfg also sets `show_static=False` so the static
  fridge/ground obey the toggles instead of being force-drawn. The cable/plug stay in both (their rod/
  plug geometry is their only visual). Use `_scratch/reports/waterhose_scaling/dump_shape_flags.py` to
  inspect the per-shape flag categorization.
