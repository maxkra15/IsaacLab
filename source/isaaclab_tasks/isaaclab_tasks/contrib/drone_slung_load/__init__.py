# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton AVBD cable-suspended drone for the FLARE waypoint-passing task."""

DRONE_SLUNG_LOAD_WANDB_PROJECT = "drone_slung_load_waypoint_flare"
"""Stable W&B project shared by every slung-load experiment."""

ENHANCED_EXPERIMENT_NAME = "drone_slung_load_waypoint_flare_enhanced_curvature_speed_v13"
"""Checkpoint namespace for curvature-aware, completion-driven path tracking."""

DIRECT_CTBR_EXPERIMENT_NAME = "drone_slung_load_waypoint_flare_direct_ctbr_pid_v14"
"""Checkpoint namespace for policy-owned collective-thrust/body-rate flight."""

DIRECT_CTBR_ROUTE_FIRST_EXPERIMENT_NAME = "drone_slung_load_waypoint_flare_direct_ctbr_body_conditioned_v16"
"""Checkpoint namespace for body-conditioned Direct-CTBR route learning."""

DRONE_DIRECT_CTBR_EXPERIMENT_NAME = "drone_waypoint_flare_direct_ctbr_body_conditioned_v2"
"""Checkpoint namespace for body-conditioned rigid-drone Direct-CTBR flight."""

DIRECT_CTBR_LONG_HORIZON_EXPERIMENT_NAME = "drone_slung_load_waypoint_flare_direct_ctbr_long_horizon_v17"
"""Checkpoint namespace for long-horizon slung-load Direct-CTBR learning."""

DRONE_DIRECT_CTBR_LONG_HORIZON_EXPERIMENT_NAME = "drone_waypoint_flare_direct_ctbr_long_horizon_v3"
"""Checkpoint namespace for long-horizon rigid-drone Direct-CTBR learning."""

DIRECT_CTBR_HARD_ROUTES_EXPERIMENT_NAME = "drone_slung_load_waypoint_flare_direct_ctbr_hard_routes_v18"
"""Checkpoint namespace for hard-route slung-load Direct-CTBR learning."""

DRONE_DIRECT_CTBR_HARD_ROUTES_EXPERIMENT_NAME = "drone_waypoint_flare_direct_ctbr_hard_routes_v4"
"""Checkpoint namespace for hard-route rigid-drone Direct-CTBR learning."""
