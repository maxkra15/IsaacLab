# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare the static output for Isaac Lab's reusable interactive-releases branch."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


def publish(source: Path, output: Path) -> str:
    """Replace the generated site and exporters in a separate publishing checkout."""
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("The output must be outside the source checkout and must not contain it")
    current = "lab" if (source / "site" / "lab").is_dir() else "interactive"
    if not (source / "site" / current).is_dir():
        raise ValueError("The source checkout has no site/lab or site/interactive gallery")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("site", "exporters"):
        destination = output / name
        if destination.is_symlink():
            destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source / name, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    site = output / "site"
    interactive = site / "interactive"
    if current == "lab":
        (site / current).rename(interactive)
    index = site / "index.html"
    index.write_text(index.read_text().replace("./v6/", "./interactive/"))
    (output / "tools").mkdir(exist_ok=True)
    shutil.copy2(source / "tools" / "publish.py", output / "tools" / "publish.py")

    # Stamp app imports and gallery navigation without renaming assets or touching released directories.
    for path in [
        interactive / "index.html",
        *interactive.glob("*.js"),
        *interactive.glob("css/*.js"),
        *interactive.glob("css/*.css"),
    ]:
        content = re.sub(r"([?&](?:rev|v)=)[\w-]+", rf"\g<1>{revision}", path.read_text())
        content = re.sub(r"(const revision = ')[\w-]+(';)", rf"\g<1>{revision}\2", content)
        content = content.replace(
            "https://github.com/maxkra15/isaaclab-browser-demos/tree/pages/exporters",
            "https://github.com/isaac-sim/IsaacLab/tree/interactive-releases/exporters",
        )
        content = content.replace("Experimental preview · ", "")
        path.write_text(content)

    for path in (interactive / "browser_demos").rglob("manifest.json"):
        manifest = json.loads(path.read_text())
        build = manifest.get("build", {})
        changed = any(name in build for name in ("modules", "apicOperations"))
        for name in ("modules", "apicOperations"):
            build.pop(name, None)
        if changed:
            path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")

    readme = interactive / "README.md"
    content = readme.read_text().replace("# Interactive lab", "# Interactive demos")
    content = content.replace(
        "Personal experimental preview: <https://maxkra15.github.io/isaaclab-browser-demos/lab/>.\n"
        "It reuses the released gallery and assets. Ant is hidden here; versioned releases\n"
        "remain immutable. Only the selected simulation is loaded.",
        "Hosted gallery: <https://isaac-sim.github.io/IsaacLab/demos/interactive/>.\n"
        "The gallery reuses the existing static assets. Only the selected simulation is loaded.",
    )
    content = (
        content.replace("site/lab/", "site/interactive/")
        .replace("Preparation for this preview", "Scene preparation")
        .replace("existing preview profiles", "existing capture profiles")
        .replace("the lab viewer", "the viewer")
    )
    content = content.replace(
        "Use a new asset path and update the module and gallery-link revisions when replacing\n"
        "a bundle, so cached files do not mix with a new simulation. Remove unused bundles\n"
        "from the mutable preview; retain published version directories.",
        "Replace current bundles at their existing paths and regenerate the publishing checkout.\n"
        "The publishing script stamps the source commit into cache revisions. Retain the\n"
        "existing compatibility directories.",
    )
    readme.write_text(content)
    readme = output / "exporters" / "README.md"
    readme.write_text(readme.read_text().replace("site/lab/", "site/interactive/"))

    (output / "README.md").write_text(
        "<!--\n"
        "Copyright (c) 2026, The Isaac Lab Project Developers "
        "(https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).\n"
        "All rights reserved.\n\nSPDX-License-Identifier: BSD-3-Clause\n-->\n\n"
        "# Isaac Lab interactive demos\n\n"
        "Static browser simulations and exporter source published from the reusable\n"
        "`interactive-releases` branch. Physics and policy inference run locally in the browser.\n\n"
        "- [Hosted gallery](https://isaac-sim.github.io/IsaacLab/demos/interactive/)\n"
        "- [Scene details and rebuilding](site/interactive/README.md)\n"
        "- [Exporters and toolchain](exporters/README.md)\n\n"
        f"Generated from source commit `{revision}`. The internal Newton Web compiler is\n"
        "required for new physics captures; it is not included here. Serving reviewed output\n"
        "requires only static files.\n\n"
        "## Update\n\n"
        "In the source project, commit the reviewed changes and run:\n\n"
        "```bash\npython tools/publish.py --output /path/to/interactive-releases-checkout\n```\n\n"
        "Review, commit and push that checkout to the same `interactive-releases` branch.\n"
        "Keep existing compatibility directories unchanged. No new release branch, tag or\n"
        "numbered gallery directory is required. The docs PR pins the resulting official\n"
        "commit archive and its SHA256; documentation serves the stable `/demos/interactive/` path.\n\n"
        "Archive URL: `https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/<publishing-commit>`.\n"
    )
    return revision


def main() -> None:
    """Prepare generated files without committing or publishing them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Separate interactive-releases checkout")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    output = args.output.expanduser().resolve()
    try:
        revision = publish(source, output)
    except ValueError as error:
        parser.error(str(error))
    print(f"Prepared {output} from {revision}; review and commit the generated files")


if __name__ == "__main__":
    main()
