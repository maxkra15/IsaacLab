# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit and validate compact recurrent coupled dynamics from real episode shards.

This learner imports Torch and the standard library only. It never initializes
Isaac Lab, Newton, a simulator, or a renderer. Checkpoint selection uses held-out
reset groups; the final test split is untouched by training and model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .dataset import EpisodeWindows, fit_normalizers, load_episodes, task_specs
from .models import CoupledWorldModel, WorldModelConfig, checkpoint_payload, load_checkpoint, rollout_loss


def _save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate_models(models, windows, *, task_id, batch_size, batches, device):
    """Report masked autoregressive prediction and persistence-baseline errors."""
    windows.generator.manual_seed(104729)
    totals = {
        key: 0.0 for key in ("loss", "observation_squared_error", "persistence_squared_error", "reward_squared_error")
    }
    state_elements = reward_elements = 0.0
    for _ in range(batches):
        batch = windows.sample(batch_size, device=device)
        for model in models:
            loss, _ = rollout_loss(model, batch, task_id=task_id)
            totals["loss"] += float(loss)
            hidden = model.encode_batch(batch, task_id=task_id)
            observation, critic = batch["observation"][:, 0], batch["critic_observation"][:, 0]
            initial_observation = observation
            for step in range(batch["action"].shape[1]):
                result = model.step(
                    observation, critic, batch["action"][:, step], batch["context"], hidden, task_id=task_id
                )
                observation, critic, hidden = result["observation"], result["critic_observation"], result["hidden"]
                mask = batch["transition_mask"][:, step]
                target = batch["observation"][:, step + 1]
                totals["observation_squared_error"] += float(((observation - target).square() * mask[:, None]).sum())
                totals["persistence_squared_error"] += float(
                    ((initial_observation - target).square() * mask[:, None]).sum()
                )
                totals["reward_squared_error"] += float(
                    ((result["reward"] - batch["reward"][:, step]).square() * mask).sum()
                )
                state_elements += float(mask.sum()) * observation.shape[-1]
                reward_elements += float(mask.sum())
    return {
        "loss": totals["loss"] / (batches * len(models)),
        "observation_rmse": (totals["observation_squared_error"] / max(1, state_elements)) ** 0.5,
        "persistence_observation_rmse": (totals["persistence_squared_error"] / max(1, state_elements)) ** 0.5,
        "reward_rmse": (totals["reward_squared_error"] / max(1, reward_elements)) ** 0.5,
        "valid_transition_count": reward_elements / len(models),
    }


def _initialize(model, source):
    # Task adapters may change dimensionality; shared dynamics keeps its identity.
    model.particle_encoder.network.load_state_dict(source.particle_encoder.network.state_dict())
    model.rigid_encoder.network.load_state_dict(source.rigid_encoder.network.state_dict())
    model.transition.load_state_dict(source.transition.state_dict())
    source_specs = {spec.task_id: spec for spec in source.config.tasks}
    for spec in model.config.tasks:
        source_spec = source_specs.get(spec.task_id)
        if source_spec is not None and (
            spec.observation_dim,
            spec.critic_observation_dim,
            spec.action_dim,
        ) != (source_spec.observation_dim, source_spec.critic_observation_dim, source_spec.action_dim):
            raise ValueError(f"Transferred checkpoint has an incompatible native interface for task {spec.task_id}")
        if source_spec is None or model.config.physical_decoder != source.config.physical_decoder:
            continue
        target_contract, source_contract = asdict(spec), asdict(source_spec)
        if model.config.physical_decoder == "none":
            for key in (
                "mass_fraction_indices",
                "quaternion_starts",
                "quaternion_order",
                "constant_observation_indices",
                "zero_observation_indices",
                "typed_action_observation_start",
                "typed_elapsed_time_index",
                "typed_elapsed_time_increment",
                "typed_countdown_index",
                "typed_countdown_decrement",
            ):
                target_contract.pop(key)
                source_contract.pop(key)
        if source_contract == target_contract:
            model.task(spec.task_id).load_state_dict(source.task(spec.task_id).state_dict())


def train(args):
    """Train independently initialized members and retain the best validation checkpoint."""
    if args.steps < 1 or args.ensemble < 1 or args.batch_size < 1 or args.horizon < 1:
        raise ValueError("Training steps, ensemble count, batch size and horizon must be positive")
    torch.set_num_threads(args.threads)
    episodes, provenance = [], []
    for directory in args.data:
        root = Path(directory)
        episodes.extend(load_episodes(root))
        provenance.append(
            {
                "path": str(root.resolve()),
                "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            }
        )
    if args.task:
        episodes = [episode for episode in episodes if episode["task_id"] in args.task]
    if not episodes:
        raise ValueError("No requested task episodes were found")
    groups = {}
    for episode in episodes:
        key = (episode["task_id"], episode["reset_group"])
        if key in groups and groups[key] != episode["split"]:
            raise ValueError(f"Combined datasets leak reset group {key} between splits")
        groups[key] = episode["split"]
    specs = task_specs(episodes)
    config = WorldModelConfig(
        specs,
        hidden_dim=args.hidden_dim,
        entity_dim=args.entity_dim,
        material_transform=args.material_transform,
        physical_decoder=args.physical_decoder,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise FileExistsError(f"Refusing to overwrite a trained model directory: {output}")
    (output / "split.json").write_text(
        json.dumps(
            [
                {
                    "task_id": episode["task_id"],
                    "reset_group": episode["reset_group"],
                    "episode_id": episode["episode_id"],
                    "split": episode["split"],
                    "source_file": episode["source_file"],
                }
                for episode in episodes
            ],
            indent=2,
        )
        + "\n"
    )
    metadata = {}
    for spec in specs:
        episode = next(episode for episode in episodes if episode["task_id"] == spec.task_id)
        recorded = {**episode.get("metadata", {}), **episode}
        metadata[spec.task_id] = {
            "context_names": recorded.get("context_names", []),
            "dt": recorded.get("dt"),
            "coordinate_frame": recorded.get("coordinate_frame", "environment_local"),
            "quaternion_order": recorded.get("quaternion_order", "wxyz"),
            "observation_names": recorded.get("observation_names", []),
            "action_names": recorded.get("action_names", []),
            "controller_names": recorded.get("controller_names", []),
            "control_mode": recorded.get("control_mode"),
            "action_contract": recorded.get("action_contract"),
        }
    train_episodes = [episode for episode in episodes if episode["split"] == "train"]
    windows = {
        spec.task_id: {
            split: EpisodeWindows(
                episodes,
                task_id=spec.task_id,
                split=split,
                horizon=args.horizon + (args.burn_in if split == "train" else 0),
                seed=args.seed,
            )
            for split in ("train", "validation")
        }
        for spec in specs
    }
    sources = None
    if args.initialize_from:
        sources, _ = load_checkpoint(args.initialize_from, device="cpu")
        if sources[0].config.hidden_dim != config.hidden_dim or sources[0].config.entity_dim != config.entity_dim:
            raise ValueError("Transferred models must have the same shared hidden/entity dimensions")
        if sources[0].config.material_transform != config.material_transform:
            raise ValueError("Transferred models must use the same explicit material transform")
    models, optimizers = [], []
    for index in range(args.ensemble):
        torch.manual_seed(args.seed + index * 1009)
        model = CoupledWorldModel(config)
        fit_normalizers(model, train_episodes)
        if sources:
            _initialize(model, sources[index % len(sources)])
        if args.freeze_core:
            for module in (model.particle_encoder.network, model.rigid_encoder.network, model.transition):
                module.requires_grad_(False)
        model.to(args.device)
        models.append(model)
        optimizers.append(
            torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=1e-5
            )
        )
    run = {
        "arguments": vars(args),
        "config": asdict(config),
        "data": provenance,
        "task_metadata": metadata,
        "episode_counts": {
            split: sum(e["split"] == split for e in episodes) for split in ("train", "validation", "test")
        },
        "scope": "Compact observation surrogate; no full particle decoder or claim of physical-force accuracy",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("models.py", "dataset.py", "train_world_model.py")
        },
        "normalization_provenance": {
            "shared_entity_and_material": "Fitted exclusively on this run's training episodes",
            "task_heads": (
                "Matching task heads retain source-checkpoint normalizers; new task heads use current training episodes"
                if args.initialize_from
                else "Fitted exclusively on this run's training episodes"
            ),
            "source_checkpoint_sha256": (
                hashlib.sha256(Path(args.initialize_from).read_bytes()).hexdigest() if args.initialize_from else None
            ),
        },
        "split_groups": {
            spec.task_id: {
                split: sorted(
                    {
                        episode["reset_group"]
                        for episode in episodes
                        if episode["task_id"] == spec.task_id and episode["split"] == split
                    }
                )
                for split in ("train", "validation", "test")
            }
            for spec in specs
        },
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    started = time.perf_counter()
    best = float("inf")
    with (output / "metrics.jsonl").open("w", buffering=1) as log:
        for step in range(1, args.steps + 1):
            spec = specs[(step - 1) % len(specs)]
            train_losses = []
            for model, optimizer in zip(models, optimizers):
                model.train()
                batch = windows[spec.task_id]["train"].sample(args.batch_size, device=args.device)
                loss, _ = rollout_loss(model, batch, task_id=spec.task_id, burn_in=args.burn_in)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at step {step}, task {spec.task_id}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"Nonfinite training gradient at step {step}")
                optimizer.step()
                train_losses.append(float(loss.detach()))
            if step == 1 or step % args.evaluate_every == 0 or step == args.steps:
                for model in models:
                    model.eval()
                validation = {
                    item.task_id: evaluate_models(
                        models,
                        windows[item.task_id]["validation"],
                        task_id=item.task_id,
                        batch_size=args.batch_size,
                        batches=args.validation_batches,
                        device=args.device,
                    )
                    for item in specs
                }
                score = sum(metrics["loss"] for metrics in validation.values()) / len(validation)
                row = {
                    "step": step,
                    "seconds": time.perf_counter() - started,
                    "training_loss": sum(train_losses) / len(train_losses),
                    "validation": validation,
                }
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                payload = checkpoint_payload(
                    models,
                    step=step,
                    validation=validation,
                    run=run,
                    optimizer_states=[optimizer.state_dict() for optimizer in optimizers],
                )
                _save(output / "last.pt", payload)
                if score < best:
                    best = score
                    _save(output / "best.pt", payload)
    (output / "COMPLETE.json").write_text(
        json.dumps(
            {"steps": args.steps, "best_validation_loss": best, "seconds": time.perf_counter() - started}, indent=2
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", nargs="+", required=True, help="Recorded dataset directories, each containing manifest.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", nargs="+", help="Train only these task IDs")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--ensemble", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--burn-in",
        type=int,
        default=0,
        help="Optional teacher-forced warmup; zero matches direct snapshot-seeded imagination",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--entity-dim", type=int, default=64)
    parser.add_argument("--material-transform", choices=("signed_log1p", "identity"), default="signed_log1p")
    parser.add_argument(
        "--physical-decoder",
        choices=("none", "typed"),
        default="none",
        help="Opt-in conservative mass transport and quaternion manifold integration",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initialize-from", help="Warm-start matching task heads and shared dynamics from a trusted checkpoint"
    )
    parser.add_argument(
        "--freeze-core", action="store_true", help="Fit task adapters while retaining shared dynamics weights"
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
