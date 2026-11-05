# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit camera/environment correspondence using a frozen Cartpole RGB harness.

Run in each backend's interpreter with ``CUDA_VISIBLE_DEVICES`` selecting an idle
GPU. The script reads an existing measured configuration, uses an isolated USD
cache, and writes validation evidence without changing measured results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import traceback
from pathlib import Path


def audit(config_path: Path, harness_dir: Path, genesis_path: Path, output: Path) -> dict:
    """Check isolated reset images and physics-driven image changes at N=512."""
    config = json.loads(config_path.read_text())
    if (config["task"], config["vision"], config["num_envs"]) != ("cartpole", "rgb", 512):
        raise ValueError("Use a measured Cartpole RGB configuration with exactly 512 environments")
    excluded = {"report", "export_figures", "assemble_report", "validate_adapter"}
    fingerprint = hashlib.sha256(
        b"".join(
            path.name.encode() + path.read_bytes()
            for path in sorted(harness_dir.glob("*.py"))
            if path.stem not in excluded
        )
    ).hexdigest()
    if fingerprint != config["provenance"]["harness_sha256"]:
        raise ValueError("The explicitly selected harness does not match the measured configuration")
    source = harness_dir.parents[2] / "source"
    sys.path[:0] = [str(harness_dir), *(str(path) for path in source.iterdir() if path.is_dir())]
    if config["backend"] == "genesis":
        sys.path.insert(0, str(genesis_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Avoid races with an active driver's robot conversion cache.
    config["asset_cache_dir"] = str(output.parent / "usd_cache")
    result = {
        "schema_version": 1,
        "kind": "camera_scale_audit",
        "status": "failed",
        "config_reference": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "harness_sha256": fingerprint,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "backend": config["backend"],
        "num_envs": config["num_envs"],
        "target_env": 511,
        "warmup_physics_steps": 16,
        "warmup_renders": 2,
        "isolated_asset_cache": config["asset_cache_dir"],
        "checks": {},
    }
    adapter = None
    try:
        import torch

        torch.set_num_threads(config["cpu_threads"])
        torch.manual_seed(config["seed"])
        module = importlib.import_module(config["adapter_module"])
        adapter = module.Adapter(config)
        result["adapter_path"] = module.__file__
        result["device_uuid"] = str(torch.cuda.get_device_properties(config["device"]).uuid)
        target = torch.tensor([511], device=config["device"], dtype=torch.long)
        actions = torch.zeros((512, 1), device=config["device"])
        adapter.reset()
        adapter.apply_action(actions)
        for _ in range(16):
            adapter.physics_step()
        for _ in range(2):
            adapter.render()
        adapter.synchronize()
        if hasattr(adapter, "diagnostics"):
            diagnostics = adapter.diagnostics()
            result["warmup_diagnostics"] = diagnostics
            result["checks"]["graph_after_warmup"] = {"passed": diagnostics["graph_active"]}

        before = adapter.render().clone()
        adapter.reset(target, joint_pos=torch.tensor([[1.0, 0.6]], device=config["device"]))
        after = adapter.render().clone()
        adapter.synchronize()
        result["checks"]["selected_reset_correspondence"] = _compare(before, after, output.parent, "reset")
        result["checks"]["selected_reset_correspondence"]["target_joint_pos"] = [1.0, 0.6]

        adapter.reset()
        adapter.apply_action(actions)
        for _ in range(4):
            adapter.physics_step()
        before = adapter.render().clone()
        joint_before = adapter.state()["joint_pos"].clone()
        actions[511, 0] = 20.0
        adapter.apply_action(actions)
        for _ in range(32):
            adapter.physics_step()
        after = adapter.render().clone()
        adapter.synchronize()
        state = adapter.state()
        joint_delta = (state["joint_pos"] - joint_before).abs().amax(-1)
        changed_states = (joint_delta > 1e-5).nonzero().flatten().tolist()
        cart_displacement = (state["joint_pos"][511, 0] - joint_before[511, 0]).item()
        result["checks"]["selected_force_moves_one_environment"] = {
            "passed": changed_states == [511] and cart_displacement > 0.05,
            "changed_state_indices": changed_states,
            "target_cart_displacement_m": cart_displacement,
            "force_n": 20.0,
            "physics_steps": 32,
        }
        result["checks"]["physics_render_freshness"] = _compare(before, after, output.parent, "physics")
        if hasattr(adapter, "diagnostics"):
            diagnostics = adapter.diagnostics()
            result["final_diagnostics"] = diagnostics
            result["checks"]["no_capacity_overflow_at_end"] = {
                "passed": not any(diagnostics["fatal_capacity_flags_at_end"])
            }
        result["status"] = "ok" if all(item["passed"] for item in result["checks"].values()) else "failed"
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if adapter is not None:
            adapter.close()
    return result


def _compare(before, after, directory: Path, phase: str) -> dict:
    """Record which cameras changed and save two representative environment views."""
    import torch
    from PIL import Image

    contract = (
        tuple(before.shape) == tuple(after.shape) == (512, 64, 64, 3)
        and before.dtype == after.dtype == torch.uint8
        and before.is_cuda
        and after.is_cuda
    )
    if not contract:
        return {"passed": False, "before_shape": list(before.shape), "after_shape": list(after.shape)}
    difference = (after.float() - before.float()).abs().flatten(1).mean(-1)
    # Mean absolute uint8 pixel difference; ignore subpixel-sized roundoff.
    changed = (difference > 0.05).nonzero().flatten().tolist()
    unselected = difference[:511].max().item()
    samples = []
    for label, frames in (("before", before), ("after", after)):
        for index in (0, 511):
            name = f"{phase}_{label}_env{index}.png"
            Image.fromarray(frames[index].cpu().numpy()).save(directory / name)
            samples.append(name)
    return {
        "passed": changed == [511],
        "changed_image_indices": changed,
        "target_mean_absolute_pixel_difference": difference[511].item(),
        "maximum_unselected_mean_absolute_pixel_difference": unselected,
        "pixel_difference_threshold": 0.05,
        "samples": samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--harness_dir", type=Path, required=True)
    parser.add_argument("--genesis_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config, args.harness_dir.resolve(), args.genesis_path.resolve(), args.output)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "ok" else 1)
