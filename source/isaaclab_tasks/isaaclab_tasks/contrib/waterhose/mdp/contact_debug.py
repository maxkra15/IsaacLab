# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional per-step contact logging for the waterhose coupled demo.

Set ``WATERHOSE_DEBUG_CONTACTS`` to print the active Newton contacts while the demo or teleop runs:

* ``1`` / ``on`` -- log a one-line summary only when the set of contacting shape pairs changes
  (a contact forms or breaks). This is the readable default for watching a grasp/insert arc.
* ``all`` / ``every`` -- log every step (verbose).

For coupled solvers the manager-level :meth:`NewtonManager.get_contacts` is empty -- the contacts live
in each coupled entry's collision pipeline and in the proxy-coupling pipeline. This module reads both:

* per-entry pipelines (``mjc`` -> robot vs. housing/floor, ``vbd`` -> cable/plug vs. socket/housing), and
* the proxy pipeline(s) (``proxy mjc->vbd`` -> the gripper-finger grip on the deformable plug),

and prints each contacting pair by a short, categorized label. Contact shape ids are global, so one
label table built from the main model maps every source. The logging is gated behind the env var, so it
adds no overhead when unset.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_MODE = os.environ.get("WATERHOSE_DEBUG_CONTACTS", "").strip().lower()
_ENABLED = _MODE not in {"", "0", "false", "no", "off"}
_EVERY_STEP = _MODE in {"all", "every"}


def _category(label: str) -> str:
    """Map a raw shape/body label to a short, readable contact-debug name."""
    low = label.lower()
    leaf = label.rstrip("/").rsplit("/", 1)[-1] if "/" in label else label
    if "bodycollision" in low:
        return "housing"
    if "socketcollision" in low:
        return "socket"
    if "leftfinger" in low:
        return "finger_L"
    if "rightfinger" in low:
        return "finger_R"
    if "plug" in low:
        return "plug"
    if "anchor" in low:
        return "anchor"
    if "cable" in low:
        return "cable"
    if "floor" in low:
        return "floor"
    if any(tok in low for tok in ("gripper", "arm", "torso", "head", "robot")):
        return f"robot:{leaf}"
    return leaf


class _ContactLogger:
    """Reads the coupled solver's per-entry and proxy contact buffers and logs active pairs."""

    def __init__(self) -> None:
        self._labels: list[str] | None = None
        self._printed_shape_materials = False
        self._last_signature: frozenset | None = None
        self._step = 0

    def _ensure_labels(self, model) -> list[str]:
        if self._labels is not None:
            return self._labels
        shape_body = model.shape_body.numpy()
        body_label = [str(x) for x in model.body_label]
        shape_label = [str(x) for x in model.shape_label] if getattr(model, "shape_label", None) is not None else None
        labels = []
        for sid in range(len(shape_body)):
            bid = int(shape_body[sid])
            raw_shape_label = shape_label[sid] if shape_label is not None else ""
            # The connector is a compound shape on the cable-head body. Prefer its
            # shape label so contact logs do not misreport plug contacts as cable contacts.
            if "waterhose_connector" in raw_shape_label.lower():
                labels.append("plug")
            elif bid >= 0:
                labels.append(_category(body_label[bid]))
            elif shape_label is not None:
                labels.append(_category(raw_shape_label))
            else:
                labels.append(f"shape{sid}")
        self._labels = labels
        return labels

    def _print_shape_materials(self, model, labels: list[str]) -> None:
        """Print the effective grip-shape contact inputs once per debug run."""
        if self._printed_shape_materials:
            return
        self._printed_shape_materials = True
        mu = model.shape_material_mu.numpy()
        margin = model.shape_margin.numpy()
        gap = model.shape_gap.numpy()
        rows = []
        for shape_id, label in enumerate(labels):
            if label in {"finger_L", "finger_R", "plug"}:
                rows.append(
                    f"{label}[{shape_id}]: mu={float(mu[shape_id]):.3g} "
                    f"margin={1.0e3 * float(margin[shape_id]):.3g}mm "
                    f"gap={1.0e3 * float(gap[shape_id]):.3g}mm"
                )
        print(f"[waterhose contacts] grip shapes: {', '.join(rows)}", flush=True)

    def _sources(self, solver):
        """Yield ``(name, Contacts)`` for every contact buffer the coupled solver exposes."""
        from isaaclab_newton.physics import NewtonManager

        # Manager-level contacts are populated when the current coupler needs the outer Newton
        # collision pipeline. This is the public buffer ``get_contacts()`` returns.
        manager_contacts = NewtonManager.get_contacts()
        if manager_contacts is not None:
            yield "manager (get_contacts)", manager_contacts
        # Proxy-coupling pipelines (e.g. the gripper grip on the deformable plug).
        for key in list(getattr(solver, "_proxy_collision_configs", {})):
            getter = getattr(solver, "get_proxy_contacts", None)
            contacts = getter(key[0], key[1]) if getter is not None else None
            if contacts is not None:
                yield f"proxy {key[0]}->{key[1]}", contacts
        # Per-entry collision pipelines.
        try:
            names = solver.entry_names()
        except Exception:  # noqa: BLE001
            names = ()
        for name in names:
            try:
                sub = solver.solver(name)
            except Exception:  # noqa: BLE001
                continue
            contacts = getattr(sub, "_contacts", None)
            if contacts is not None:
                yield f"entry {name}", contacts

    def __call__(self) -> None:
        from isaaclab_newton.physics import NewtonManager

        solver = getattr(NewtonManager, "_solver", None)
        model = NewtonManager.get_model()
        if solver is None or model is None:
            return
        self._step += 1
        labels = self._ensure_labels(model)
        self._print_shape_materials(model, labels)
        n_shapes = len(labels)

        from collections import Counter

        # (source, kind, pairA, pairB) -> count. ``source`` is included so a contact handled by two
        # pipelines (e.g. gripper vs. housing in both the MJWarp entry and the proxy) shows up distinctly.
        tallies: Counter = Counter()
        for source, contacts in self._sources(solver):
            count_arr = getattr(contacts, "rigid_contact_count", None)
            rc = int(count_arr.numpy()[0]) if count_arr is not None else 0
            if rc:
                s0 = contacts.rigid_contact_shape0.numpy()[:rc]
                s1 = contacts.rigid_contact_shape1.numpy()[:rc]
                for a, b in zip(s0, s1):
                    a, b = int(a), int(b)
                    if 0 <= a < n_shapes and 0 <= b < n_shapes:
                        pair = tuple(sorted((labels[a], labels[b])))
                        tallies[(source, "rigid", pair[0], pair[1])] += 1
            soft_arr = getattr(contacts, "soft_contact_count", None)
            sc = int(soft_arr.numpy()[0]) if soft_arr is not None else 0
            if sc:
                shapes = contacts.soft_contact_shape.numpy()[:sc]
                for x in shapes:
                    x = int(x)
                    if 0 <= x < n_shapes:
                        tallies[(source, "soft", "particle", labels[x])] += 1

        signature = frozenset(tallies)  # the set of (source, kind, pair) keys, ignoring counts
        if not _EVERY_STEP and signature == self._last_signature:
            return
        self._last_signature = signature

        if not tallies:
            print(f"[waterhose contacts] step {self._step}: no active contacts", flush=True)
            return

        by_source: dict[str, list[str]] = {}
        for (source, kind, a, b), n in sorted(tallies.items()):
            by_source.setdefault(source, []).append(f"{a}<->{b} x{n}")
        summary = " | ".join(f"{source}: {', '.join(items)}" for source, items in by_source.items())
        print(f"[waterhose contacts] step {self._step}: {summary}", flush=True)


_LOGGER = _ContactLogger() if _ENABLED else None


def log_contacts_if_enabled() -> None:
    """Log active Newton contacts when ``WATERHOSE_DEBUG_CONTACTS`` is set; a no-op otherwise."""
    if _LOGGER is None:
        return
    try:
        _LOGGER()
    except Exception:  # noqa: BLE001 -- debug aid must never break a run
        logger.exception("[waterhose contacts] contact logging failed; disabling for this run")
        _disable()


def _disable() -> None:
    global _LOGGER
    _LOGGER = None
