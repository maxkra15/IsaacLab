# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task capability checks for Isaac Lab Mimic command-line workflows."""


def validate_skillgen_task_support(task_name: str, env_cfg: object) -> None:
    """Raise a clear error when a task explicitly does not support SkillGen.

    Tasks opt out by defining a non-empty ``skillgen_unsupported_reason`` on
    their environment configuration. Configurations without that attribute
    retain the existing SkillGen behavior.
    """
    reason = getattr(env_cfg, "skillgen_unsupported_reason", None)
    if reason is None:
        return
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("skillgen_unsupported_reason must be a non-empty string when configured.")
    raise ValueError(f"--use_skillgen is not supported for task '{task_name}': {reason.strip()}")
