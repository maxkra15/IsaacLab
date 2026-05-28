# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Controller exports for the waterhose robot demo."""

from __future__ import annotations


def get_task_type_enum():
    """Return the local scripted state-machine enum."""
    from .simulation import TaskType  # noqa: PLC0415

    return TaskType
