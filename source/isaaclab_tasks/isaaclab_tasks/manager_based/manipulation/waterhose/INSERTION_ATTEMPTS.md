# Waterhose Cable Insertion Attempts

This log records the main variants tested while replacing the tail fixed joint and trying to complete `run_robot_demo.py`.

## Stable Baseline Replacement

- Replaced the `Anchor1` fixed joint tail weld with static VBD SDF capture geometry on `Cable1`.
- Added one closed terminal cup and several short through-sleeve SDF captures on the last cable segments.
- Added shape-label selection so those static SDF capture shapes are visible to the VBD coupled-solver entry.
- Result: cable remains finite with bounds O(0.1 m) during grasp and socket approach. The historical contact blow-up did not return.

## Proxy And Grip Tuning

- Kept proxy coupling effectively immovable with `mass_scale=1e3`, `collide_interval=1`, sticky contact matching.
- Added gripper base bodies to the proxy set in addition to finger bodies.
- Added coupled-manager support for global model material overrides and per-proxy shape material overrides.
- Set gripper proxy material to reference-style high friction/stiff contact: `shape_material_ke=2e5`, `shape_material_mu=3e6`, `shape_margin=0.001`.
- Slowed and capped the grip controller: force target `25 N`, close cap `-0.93`, tighten rate `0.18`.
- Result: grip feedback improved from sub-newton/noisy contact to roughly 24-35 N through `SETTLE`, and the 6-7 kN crush spike was avoided.

## State-Machine Variants

- Added reference-style post-blend finger centering correction, capped at 3 mm per frame and projected off the plug axis.
- Tried live grasp-frame socket transfer with preserved orientation.
- Tried transfer with socket-aligned flipped plug orientation.
- Tried shortening transfer and skipping the high-torque align step.
- Result: high-friction preserved-orientation transfer gets the plug near the socket neighborhood, but contact drops near the end of `APPROACH_TARGET`. At failure, `grip_force=0`, `finger_mid_err` is about 0.65 m, and the cable remains finite. The flipped-orientation transfer ejects the plug earlier. The shortened/skip-align variant can reach `DONE` only after losing the plug and then drives the cable to unphysical depths, so it was rejected.

## Current Failure Signature

- Stable candidate command:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 --vis none \
  --max_steps 1150 --settle_time 0.5 --debug_script
```

- Representative result:
  - `GRASP`: `finger_mid_err ~= 0.0013`, `grip_force ~= 0.44`, finite cable.
  - `HOLD_GRASP`: `finger_mid_err ~= 0.0018`, `grip_force ~= 24.45`, finite cable.
  - `SETTLE`: `finger_mid_err ~= 0.0018`, `grip_force ~= 34.27`, finite cable.
  - `APPROACH_TARGET`: starts held, `finger_mid_err ~= 0.0018`, `grip_force ~= 34.24`.
  - Near end of `APPROACH_TARGET`, plug reaches the socket neighborhood, then slips.
  - `ALIGN_AXES`: `finger_mid_err ~= 0.66`, `grip_force=0`, cable still finite and O(0.1 m).

## Root-Cause Evidence

The tail fixed-joint weld is no longer the observed blocker. The SDF-captured tail remains stable under the same proxy approach and grasp phases. The repeatable blocker is plug retention during the socket transfer/orientation handoff: the plug is held through grasp/settle, approaches the socket, then loses proxy contact before alignment/insertion can complete.

The most likely remaining causes are:

- Plug/gripper contact geometry does not provide mechanical capture; retention depends on friction and is lost during the long transfer and later reorientation.
- The grasp orientation is not insertion-compatible (`tip_axis_cos` remains positive around 0.3 before alignment), so the align phase must rotate a held plug substantially and tends to eject it.
- The scripted transfer continues open-loop after the plug reaches the socket neighborhood, so once slip starts the arm keeps moving away from the live plug frame.

## Scripted IK Twist/Grip Pass - June 2

- Corrected the scripted insertion-axis sign. The desired plug axis is now `insertion_dir_w`, not `-insertion_dir_w`. Runs after this change keep `tip_axis_cos` positive and improving during the turn; the earlier wrong-way twist was real.
- Added a grip lock: the gripper only tightens during `GRASP/HOLD_GRASP`, then holds the locked command through transfer/insertion. This removes the repeated post-grasp tightening behavior.
- Made `RETRACT` a phase-latched axis turn and `SETTLE` a true hold. Live moving turn targets caused runaway cable motion; phase-latched turn targets keep the cable finite.
- Changed the IK/action EE offset back to the actual right finger-link midpoint from the robot USD: `(0, 0, -0.075)` under `right_gripper_base`. The previous `-0.1045` target was about 30 mm past the finger link midpoint.
- Fixed the centering correction sign. Then tested centering in several scopes:
  - `ENGAGE + GRASP`: still dragged the plug during closure.
  - `ENGAGE` only: best conservative result.
  - `ENGAGE + APPROACH_TARGET`, capped at 2 mm/step: worsened transfer slip.
- Added a lost-grip guard so zero proxy force during `ALIGN_AXES/VERIFY_ALIGN/INSERT` holds the hand and prevents the state machine from timing out into later phases after the plug has already slipped.

Representative current run:

```bash
./isaaclab.sh -p scripts/environments/waterhose/run_robot_demo.py \
  --task Isaac-Waterhose-Coupled-v0 --vis none \
  --max_steps 1500 --settle_time 0.5 --debug_script
```

Best conservative result after the frame/sign fixes:

- `GRASP`: `finger_mid_err ~= 0.006 m`, `grip_force ~= 0.4 N`, `tip_axis_cos ~= 0.28`.
- `HOLD_GRASP`: closure itself pushes the plug/cable sideways by about 20-30 mm; `finger_mid_err ~= 0.030 m`, `grip_force ~= 34 N`, cable remains finite.
- `SETTLE`: after the corrected axis turn, `grip_force ~= 39 N`, `tip_axis_cos ~= 0.38`, cable remains finite and bounded.
- `APPROACH_TARGET`: transfer starts with nonzero force, then decays to about `13 N`.
- `ALIGN_AXES`: grip is lost before alignment/insertion, with `grip_force=0` and `finger_mid_err ~= 0.19 m`. The cable remains finite; the guard prevents verify/insert from chasing the lost plug.

Diagnostic result with transfer-only live centering:

- `SETTLE`: `grip_force ~= 96 N`, `tip_axis_cos ~= 0.38`.
- `APPROACH_TARGET`: grip still decays to about `13 N`.
- `ALIGN_AXES`: grip is again lost, with `grip_force=0` and `finger_mid_err ~= 0.22 m`.
- Conclusion: live transfer centering does not recover the slipping plug; it makes the final separation worse.

Current conclusion:

- The scripted twist sign and the post-grasp over-tightening were real bugs and are fixed.
- The cable is stable; the old tail-weld/contact explosion is not the current blocker.
- The remaining blocker is mechanical retention of the plug. With the present plug/finger geometry, closing the fingers already shoves the plug/cable laterally, and the long transfer loses contact before socket alignment. A keyed plug, gripper sleeve/cup, or other mechanical capture geometry is needed to transmit the required turn/transport reliably; more open-loop IK corrections did not solve it.
