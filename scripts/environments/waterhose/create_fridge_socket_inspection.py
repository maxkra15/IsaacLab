"""Create an Isaac Sim inspection layer for the Waterhose fridge socket.

The fridge USD has many Blender-exported collider prims without semantic names.
This helper ranks the collider meshes near the configured socket target and
writes a small USDA layer that references the fridge, promotes collision meshes
to normal visible render geometry, highlights the likely socket colliders, and
adds marker spheres for the socket and inserted plug tip.

Usage:
    ./isaaclab.sh -p scripts/environments/waterhose/create_fridge_socket_inspection.py
    ./isaaclab.sh -s source/isaaclab_tasks/isaaclab_tasks/contrib/waterhose/assets/fridge/fridge_socket_inspection.usda
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FRIDGE_DIR = (
    REPO_ROOT
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "contrib"
    / "waterhose"
    / "assets"
    / "fridge"
)
DEFAULT_FRIDGE_USD = FRIDGE_DIR / "fridge.usda"
DEFAULT_OUTPUT_USD = FRIDGE_DIR / "fridge_socket_inspection.usda"

# Fridge-local equivalents of the current task constants.
SOCKET_TARGET = (-0.259404, 0.362961, -0.262711)
INSERT_TRAVEL = 0.03
INSERT_ANGLE_RAD = math.radians(20.0)
INSERT_DIRECTION = (0.0, -math.sin(INSERT_ANGLE_RAD), math.cos(INSERT_ANGLE_RAD))
INSERTED_TIP_TARGET = tuple(SOCKET_TARGET[i] + INSERT_TRAVEL * INSERT_DIRECTION[i] for i in range(3))

CORE_SOCKET_CANDIDATES = {
    "Cable008_Collider52",
    "Cable008_Collider103",
    "Cable008_Collider151",
}
LIKELY_SOCKET_CANDIDATES = CORE_SOCKET_CANDIDATES | {
    "Cable008_Collider69",
    "Cable008_Collider81",
    "Cable008_Collider84",
    "Cable008_Collider116",
    "Cable008_Collider129",
    "Cable008_Collider132",
    "Cable008_Collider163",
    "Cable008_Collider216",
    "Cable008_Collider231",
}


@dataclass(frozen=True)
class ColliderCandidate:
    name: str
    line: int
    blender_object_name: str
    approximation: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    center: tuple[float, float, float]
    distance_to_socket: float
    distance_to_inserted_tip: float

    @property
    def score(self) -> float:
        return min(self.distance_to_socket, self.distance_to_inserted_tip)


def _parse_vec3(raw: str) -> tuple[float, float, float]:
    values = [float(part.strip()) for part in raw.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected 3-vector, got: {raw}")
    return values[0], values[1], values[2]


def _aabb_distance(point: tuple[float, float, float], low: tuple[float, float, float], high: tuple[float, float, float]) -> float:
    sq = 0.0
    for value, lo, hi in zip(point, low, high):
        if value < lo:
            sq += (lo - value) ** 2
        elif value > hi:
            sq += (value - hi) ** 2
    return math.sqrt(sq)


def _transform_extent(
    extent_min: tuple[float, float, float],
    extent_max: tuple[float, float, float],
    scale: tuple[float, float, float],
    translate: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    transformed_min = []
    transformed_max = []
    for lo, hi, s, t in zip(extent_min, extent_max, scale, translate):
        a = lo * s + t
        b = hi * s + t
        transformed_min.append(min(a, b))
        transformed_max.append(max(a, b))
    return tuple(transformed_min), tuple(transformed_max)


def parse_colliders(fridge_usd: Path) -> list[ColliderCandidate]:
    mesh_re = re.compile(r'^\s*def Mesh "(Cable008_Collider\d+)"')
    extent_re = re.compile(r"float3\[\] extent = \[\(([^)]*)\), \(([^)]*)\)\]")
    scale_re = re.compile(r"float3 xformOp:scale = \(([^)]*)\)")
    translate_re = re.compile(r"double3 xformOp:translate = \(([^)]*)\)")
    object_name_re = re.compile(r'custom string userProperties:blender:object_name = "([^"]+)"')
    approximation_re = re.compile(r'uniform token physics:approximation = "([^"]+)"')

    colliders: list[ColliderCandidate] = []
    current: dict[str, object] | None = None

    def finalize() -> None:
        if current is None or "extent_min" not in current:
            return
        extent_min = current["extent_min"]
        extent_max = current["extent_max"]
        scale = current.get("scale", (1.0, 1.0, 1.0))
        translate = current.get("translate", (0.0, 0.0, 0.0))
        bbox_min, bbox_max = _transform_extent(extent_min, extent_max, scale, translate)
        center = tuple((lo + hi) * 0.5 for lo, hi in zip(bbox_min, bbox_max))
        colliders.append(
            ColliderCandidate(
                name=current["name"],
                line=current["line"],
                blender_object_name=current.get("blender_object_name", ""),
                approximation=current.get("approximation", ""),
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                center=center,
                distance_to_socket=_aabb_distance(SOCKET_TARGET, bbox_min, bbox_max),
                distance_to_inserted_tip=_aabb_distance(INSERTED_TIP_TARGET, bbox_min, bbox_max),
            )
        )

    for line_number, line in enumerate(fridge_usd.read_text().splitlines(), start=1):
        if match := mesh_re.match(line):
            finalize()
            current = {"name": match.group(1), "line": line_number}
            continue
        if current is None:
            continue
        if match := extent_re.search(line):
            current["extent_min"] = _parse_vec3(match.group(1))
            current["extent_max"] = _parse_vec3(match.group(2))
        elif match := scale_re.search(line):
            current["scale"] = _parse_vec3(match.group(1))
        elif match := translate_re.search(line):
            current["translate"] = _parse_vec3(match.group(1))
        elif match := object_name_re.search(line):
            current["blender_object_name"] = match.group(1)
        elif match := approximation_re.search(line):
            current["approximation"] = match.group(1)

    finalize()
    return sorted(colliders, key=lambda candidate: (candidate.score, candidate.name))


def _fmt_vec(values: tuple[float, float, float]) -> str:
    return f"({values[0]:.9g}, {values[1]:.9g}, {values[2]:.9g})"


def _asset_reference_path(asset_path: Path, output_path: Path) -> str:
    rel_path = os.path.relpath(asset_path.resolve(), output_path.resolve().parent)
    return Path(rel_path).as_posix()


def _material_block(name: str, color: tuple[float, float, float], opacity: float = 1.0) -> str:
    return f"""        def Material "{name}"
        {{
            token outputs:surface.connect = </Inspection/Materials/{name}/PreviewSurface.outputs:surface>

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {_fmt_vec(color)}
                float inputs:opacity = {opacity:.3f}
                float inputs:roughness = 0.5
                token outputs:surface
            }}
        }}
"""


def _collider_over(candidate: ColliderCandidate, rank: int | None, material: str) -> str:
    rank_block = "" if rank is None else f"                    custom int inspection:rank = {rank}\n"
    return f"""                over "{candidate.name}"
                {{
                    rel material:binding = </Inspection/Materials/{material}> (
                        bindMaterialAs = "strongerThanDescendants"
                    )
                    uniform token purpose = "default"
                    token visibility = "inherited"
{rank_block}                    custom int inspection:source_line = {candidate.line}
                    custom double inspection:distance_to_socket_m = {candidate.distance_to_socket:.9g}
                    custom double inspection:distance_to_inserted_tip_m = {candidate.distance_to_inserted_tip:.9g}
                    custom string inspection:blender_object_name = "{candidate.blender_object_name}"
                    custom string inspection:physics_approximation = "{candidate.approximation}"
                }}
"""


def _marker_sphere(name: str, position: tuple[float, float, float], radius: float, material: str) -> str:
    return f"""        def Sphere "{name}"
        {{
            double radius = {radius:.6g}
            rel material:binding = </Inspection/Materials/{material}>
            double3 xformOp:translate = {_fmt_vec(position)}
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
"""


def write_inspection_usd(
    output_usd: Path,
    fridge_usd: Path,
    colliders: list[ColliderCandidate],
    selected: list[ColliderCandidate],
    *,
    hide_visuals: bool,
    only_candidates: bool,
) -> None:
    reference_path = _asset_reference_path(fridge_usd, output_usd)
    selected_ranks = {candidate.name: rank for rank, candidate in enumerate(selected, start=1)}
    selected_names = set(selected_ranks)

    material_blocks = [
        _material_block("FridgeContext", (0.32, 0.32, 0.34), 0.35),
        _material_block("AllColliders", (0.55, 0.55, 0.58), 0.75),
        _material_block("SocketCore", (0.05, 0.9, 0.15)),
        _material_block("SocketLikely", (1.0, 0.55, 0.05)),
        _material_block("SocketNearby", (0.1, 0.45, 1.0)),
        _material_block("SocketTarget", (0.0, 1.0, 0.8)),
        _material_block("InsertedTipTarget", (1.0, 0.0, 0.9)),
        _material_block("InsertPath", (1.0, 1.0, 0.1)),
    ]

    collider_blocks = []
    shown_colliders = selected if only_candidates else colliders
    for candidate in shown_colliders:
        if candidate.name in CORE_SOCKET_CANDIDATES:
            material = "SocketCore"
        elif candidate.name in LIKELY_SOCKET_CANDIDATES:
            material = "SocketLikely"
        elif candidate.name in selected_names:
            material = "SocketNearby"
        else:
            material = "AllColliders"
        collider_blocks.append(_collider_over(candidate, selected_ranks.get(candidate.name), material))

    path_markers = []
    for i in range(1, 5):
        alpha = i / 5.0
        position = tuple(SOCKET_TARGET[j] + alpha * INSERT_TRAVEL * INSERT_DIRECTION[j] for j in range(3))
        path_markers.append(_marker_sphere(f"InsertPath_{i:02d}", position, 0.004, "InsertPath"))

    if hide_visuals:
        visual_override = '            over "Visuals"\n            {\n                token visibility = "invisible"\n            }\n'
    else:
        visual_override = """            over "Visuals"
            {
                rel material:binding = </Inspection/Materials/FridgeContext> (
                    bindMaterialAs = "strongerThanDescendants"
                )
                uniform token purpose = "default"
                token visibility = "inherited"
            }
"""

    content = f"""#usda 1.0
(
    defaultPrim = "Inspection"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Inspection"
{{
    def Xform "Fridge" (
        references = @{reference_path}@</root>
    )
    {{
        over "Cable008"
        {{
{visual_override}            over "Collisions"
            {{
                uniform token purpose = "default"
                token visibility = "inherited"
{''.join(collider_blocks)}            }}
        }}
    }}

    def Scope "Materials"
    {{
{''.join(material_blocks)}    }}

    def Xform "Markers"
    {{
{_marker_sphere("SocketTarget", SOCKET_TARGET, 0.012, "SocketTarget")}{_marker_sphere("InsertedTipTarget", INSERTED_TIP_TARGET, 0.01, "InsertedTipTarget")}{''.join(path_markers)}    }}
}}
"""
    output_usd.write_text(content)


def write_report(report_path: Path, selected: list[ColliderCandidate]) -> None:
    with report_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "rank",
                "name",
                "line",
                "blender_object_name",
                "approximation",
                "distance_to_socket_m",
                "distance_to_inserted_tip_m",
                "bbox_min",
                "bbox_max",
                "center",
            ]
        )
        for rank, candidate in enumerate(selected, start=1):
            writer.writerow(
                [
                    rank,
                    candidate.name,
                    candidate.line,
                    candidate.blender_object_name,
                    candidate.approximation,
                    f"{candidate.distance_to_socket:.9g}",
                    f"{candidate.distance_to_inserted_tip:.9g}",
                    _fmt_vec(candidate.bbox_min),
                    _fmt_vec(candidate.bbox_max),
                    _fmt_vec(candidate.center),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fridge-usd", type=Path, default=DEFAULT_FRIDGE_USD)
    parser.add_argument("--output-usd", type=Path, default=DEFAULT_OUTPUT_USD)
    parser.add_argument("--top-n", type=int, default=32, help="Number of nearest collider meshes to highlight.")
    parser.add_argument("--hide-visuals", action="store_true", help="Hide the visual fridge mesh.")
    parser.add_argument(
        "--only-candidates",
        action="store_true",
        help="Show only ranked candidate colliders instead of all Cable008 collider meshes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    colliders = parse_colliders(args.fridge_usd)
    forced = {candidate.name for candidate in colliders if candidate.name in LIKELY_SOCKET_CANDIDATES}
    selected_names = [candidate.name for candidate in colliders[: args.top_n]]
    selected_names.extend(sorted(forced - set(selected_names)))
    selected_by_name = {candidate.name: candidate for candidate in colliders}
    selected = sorted((selected_by_name[name] for name in selected_names), key=lambda candidate: (candidate.score, candidate.name))

    args.output_usd.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.output_usd.with_suffix(".csv")
    write_inspection_usd(
        args.output_usd,
        args.fridge_usd,
        colliders,
        selected,
        hide_visuals=args.hide_visuals,
        only_candidates=args.only_candidates,
    )
    write_report(report_path, selected)

    print(f"Wrote inspection USD: {args.output_usd}")
    print(f"Wrote candidate report: {report_path}")
    print("Open with:")
    print(f"  ./isaaclab.sh -s {args.output_usd}")
    print("Select highlighted prims under /Inspection/Fridge/Cable008/Collisions to inspect exact mesh names.")


if __name__ == "__main__":
    main()
