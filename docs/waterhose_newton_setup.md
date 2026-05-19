# RBY1 Waterhose Newton Setup

Last verified: 2026-05-19.

## Branches

Use these branches together:

- Isaac Lab: `feat/newton-implicit-mpm`
- Newton: `pr-2848-coupled-solver-framework-latest`

The Newton checkout should be installed editable into Isaac Lab's Python
environment:

```bash
cd /home/maximiliank/Work/IsaacLab
./_isaac_sim/python.sh -m pip install -e /home/maximiliank/Work/newton
```

## External Asset Bundle

The waterhose/fridge asset is intentionally not stored in git. Restore it from
the external bundle before running the task:

```bash
cd /home/maximiliank/Work/IsaacLab
tar -xzf /home/maximiliank/Work/waterhose_asset_bundle.tar.gz
```

This restores:

```text
source/isaaclab_assets/data/Props/Waterhose/Cable008
```

The RBY1DF robot asset remains in the repository.

## Viewer Notes

The task supports both the Newton viewer and the Kit visualizer. Kit requires the
Isaac Sim app to launch before Newton/PXR-facing waterhose imports; the waterhose
launch helper handles this when `--visualizer kit` is present.

If a viewer does not open on the workstation display, set:

```bash
export DISPLAY=:1
```

## Teleoperation

SpaceMouse teleoperation with simple right end-effector controls:

```bash
cd /home/maximiliank/Work/IsaacLab
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Waterhose-RBY1DF-IK-Rel-Play-v0 \
    --num_envs 1 \
    --device cuda:1 \
    --visualizer newton \
    --teleop_device spacemouse \
    --spacemouse_mode simple \
    --sensitivity 1
```

Kit visualizer teleoperation:

```bash
cd /home/maximiliank/Work/IsaacLab
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Waterhose-RBY1DF-IK-Rel-Play-v0 \
    --num_envs 1 \
    --device cuda:0 \
    --visualizer kit \
    --teleop_device spacemouse \
    --spacemouse_mode simple \
    --sensitivity 1
```

Random-agent smoke test:

```bash
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-Waterhose-RBY1DF-IK-Rel-v0 \
    --num_envs 1 \
    --device cuda:1 \
    --visualizer newton
```

Scripted demo recording:

```bash
./isaaclab.sh -p scripts/tools/record_waterhose_scripted_demo.py \
    --task Isaac-Waterhose-RBY1DF-IK-Rel-v0 \
    --num_envs 1 \
    --device cuda:1 \
    --visualizer newton \
    --dataset_file ./datasets/waterhose_scripted.hdf5
```

Manual demo recording:

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Waterhose-RBY1DF-IK-Rel-Play-v0 \
    --device cuda:1 \
    --visualizer newton \
    --teleop_device spacemouse \
    --dataset_file ./datasets/waterhose_demos.hdf5
```

## Mimic

Mimic currently runs through the Newton viewer path only. The annotation path is
not clean yet: `scripts/imitation_learning/isaaclab_mimic/annotate_demos.py`
contains task-specific launch handling so annotation can work kit-less. This was
needed because the normal Kit launch path is not working for the waterhose task
right now.

Annotate a recorded dataset:

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Waterhose-RBY1DF-IK-Rel-Mimic-v0 \
    --device cuda:1 \
    --visualizer newton \
    --input_file ./datasets/waterhose_demos.hdf5 \
    --output_file ./datasets/waterhose_demos_annotated.hdf5 \
    --auto
```

Generate Mimic rollouts from an annotated dataset:

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-Waterhose-RBY1DF-IK-Rel-Mimic-v0 \
    --device cuda:1 \
    --visualizer newton \
    --num_envs 1 \
    --input_file ./datasets/waterhose_demos_annotated.hdf5 \
    --output_file ./datasets/waterhose_generated.hdf5
```
