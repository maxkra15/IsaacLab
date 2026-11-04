# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run exactly one measured configuration in a fresh simulator process."""

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path


def run(config: dict, output: Path) -> dict:
    """Measure startup, warmed stepping, or shared PPO and write an auditable result."""
    result = {
        "schema_version": 1,
        "config": config,
        "provenance": config.pop("provenance"),
        "status": "failed",
        "metrics": {},
    }
    adapter = None
    try:
        start = time.perf_counter()
        module = importlib.import_module(config["adapter_module"])
        import torch
        from tasks import Task

        torch.set_num_threads(config["cpu_threads"])
        torch.manual_seed(config["seed"])
        adapter = module.Adapter(config)
        task = Task(adapter, config)
        obs = task.reset()
        adapter.synchronize()
        metrics = result["metrics"]
        metrics["startup_s"] = time.perf_counter() - start
        result["backend_metadata"] = adapter.metadata
        result["provenance"]["runtime"] = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "adapter_path": module.__file__,
            "cuda_device": torch.cuda.get_device_name(),
            "source_packages": {
                name: sys.modules[name].__file__
                for name in ("isaaclab", "isaaclab_physx", "genesis", "torch")
                if name in sys.modules
            },
        }
        if config["vision"] == "rgb":
            images = task.images
            expected = (config["num_envs"], config["height"], config["width"])
            if tuple(images.shape[:3]) != expected or images.shape[-1] not in (3, 4):
                raise ValueError(f"Camera must deliver every env: {tuple(images.shape)}, expected {expected} + RGB")
            if images.device.type != "cuda":
                raise ValueError("Vision comparison requires GPU-resident camera output")
            if not bool(torch.isfinite(images.float()).all()) or images[..., :3].float().std() < 1:
                raise ValueError("Camera produced invalid or blank images")
            from PIL import Image

            Image.fromarray(images[0, ..., :3].to(torch.uint8).cpu().numpy()).save(output.parent / "camera.png")
            result["camera_sample"] = "camera.png"
        actions = torch.zeros((config["num_envs"], task.action_dim), device=config["device"])
        adapter.synchronize()
        first = time.perf_counter()
        task.step(actions)
        adapter.synchronize()
        metrics["first_step_s"] = time.perf_counter() - first
        warmup = time.perf_counter()
        for _ in range(config["warmup_steps"]):
            task.step(actions)
        adapter.synchronize()
        metrics["warmup_s"] = time.perf_counter() - warmup
        task.generator.manual_seed(config["seed"] + 1000)
        if config["mode"] == "learning":
            from ppo import train

            metrics.update(train(task, config, output.parent))
        elif config["mode"] == "physics":
            # Restore active contact/initial states between short windows. Neither
            # resets, state queries nor rewards are included in physics timing.
            windows = []
            remaining = config["steps"]
            while remaining:
                count = min(remaining, 100)
                task.reset()
                adapter.apply_action(task.nominal.expand(task.n, -1) if task.walk else actions)
                adapter.synchronize()
                begin = time.perf_counter()
                for _ in range(count):
                    adapter.physics_only_step()
                adapter.synchronize()
                windows.append(time.perf_counter() - begin)
                adapter.refresh_state()
                remaining -= count
            elapsed = sum(windows)
            throughput = config["steps"] * task.n / elapsed
            metrics.update(
                {
                    "elapsed_s": elapsed,
                    "physics_steps_per_s": throughput,
                    "rtf": throughput * config["dt"],
                    "window_elapsed_s": windows,
                }
            )
        else:
            # Seeded actions exercise controllers, state, reward, resets, and optional RGB.
            task.reset()
            generator = torch.Generator(device=config["device"]).manual_seed(config["seed"] + 42)
            adapter.synchronize()
            begin = time.perf_counter()
            for _ in range(config["steps"]):
                actions.uniform_(-0.25, 0.25, generator=generator)
                obs, _, _, _ = task.step(actions)
            adapter.synchronize()
            elapsed = time.perf_counter() - begin
            throughput = config["steps"] * task.n / elapsed
            metrics.update(
                {
                    "elapsed_s": elapsed,
                    "env_steps_per_s": throughput,
                    "physics_steps_per_s": throughput * config["decimation"],
                    "rtf": throughput * config["decimation"] * config["dt"],
                }
            )
            if config["vision"] == "rgb":
                # Partial resets can add frames; count only guaranteed control frames.
                metrics["render_frames_per_s"] = throughput
        state = adapter.state()
        if not all(bool(torch.isfinite(value).all()) for value in state.values()):
            raise FloatingPointError("Final dynamics state is non-finite")
        metrics["final_root_height_m"] = state["root_pos"][:, 2].mean().item()
        metrics["max_root_displacement_m"] = task.max_root_displacement.item()
        if config["vision"] == "rgb":
            # Conservative distance bound for all recorded actor trajectories:
            # Far plane 20 m + camera offset < 6.2 m + robot extent < 5 m.
            separation = config["env_spacing"] - 2 * metrics["max_root_displacement_m"]
            if separation <= 32:
                raise ValueError("Recorded trajectories can expose neighboring environments to cameras")
        metrics["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["status"] = "ok"
    except NotImplementedError as error:
        result.update(status="unsupported", error=str(error))
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        # Persist evidence before native simulator teardown can fail or hang.
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if adapter is not None:
            adapter.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), args.output)
    raise SystemExit(0 if result["status"] == "ok" else 1)
