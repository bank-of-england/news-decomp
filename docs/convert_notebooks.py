"""Convert Marimo apps in examples/ to Markdown for Zensical.

Run this script whenever a Marimo example changes:
    python docs/convert_notebooks.py

Check that the committed Markdown exports are current:
    python docs/convert_notebooks.py --check

Produces a Markdown file for each ``*_marimo.py`` app in
``docs/notebooks/`` using Marimo.
"""

import argparse
import subprocess
import sys
from pathlib import Path

APPS_DIR = Path(__file__).parent.parent / "examples"
OUTPUT_DIR = Path(__file__).parent / "notebooks"
REPO_ROOT = APPS_DIR.parent


def convert(app: Path) -> None:
    """Export one Marimo app as a Markdown documentation page."""
    print(f"Converting {app.name} ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "md",
            str(app),
            "--output",
            str(OUTPUT_DIR / f"{app.stem.removesuffix('_marimo')}.md"),
            "--force",
        ],
        check=True,
    )


def main() -> int:
    """Update the Markdown exports for the Marimo examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if conversion changes a committed Markdown export",
    )
    args = parser.parse_args()

    apps = sorted(APPS_DIR.glob("*_marimo.py"))
    if not apps:
        print("No Marimo apps found in", APPS_DIR)
        return 0

    OUTPUT_DIR.mkdir(exist_ok=True)
    for app in apps:
        convert(app)

    if args.check:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--exit-code",
                "--",
                str(OUTPUT_DIR.relative_to(REPO_ROOT)),
            ],
            check=False,
            cwd=REPO_ROOT,
        )
        if result.returncode:
            print(
                "Notebook Markdown exports are stale. "
                "Run `python docs/convert_notebooks.py` and commit the changes."
            )
            return result.returncode

    print("\nDone. Re-run this script whenever a Marimo app changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
