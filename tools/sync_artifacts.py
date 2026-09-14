"""Copy pipeline artifacts into the web application's public directory.

Run:
    python3 tools/sync_artifacts.py

The site fetches artifacts at runtime rather than bundling them, so each project page
downloads only its own data. This script is the one place that knows how to get the JSON
from where the pipelines write it to where the site serves it from.

It also reports total payload per project, because artifact size is a real constraint on a
static site and it is easy to let an exploratory dump grow to several megabytes without
noticing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
TARGET = ROOT / "web" / "public" / "data"


def main() -> None:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    rows = []

    for project_dir in sorted(PROJECTS.iterdir()):
        artifacts = project_dir / "artifacts"
        if not artifacts.is_dir():
            continue

        destination = TARGET / project_dir.name
        destination.mkdir(parents=True, exist_ok=True)

        project_bytes = 0
        files = sorted(artifacts.glob("*.json"))
        for source in files:
            # Validate as we copy: a malformed artifact should fail here, loudly, rather
            # than as an opaque fetch error in the browser.
            try:
                json.loads(source.read_text())
            except json.JSONDecodeError as error:
                raise SystemExit(f"{source} is not valid JSON: {error}")
            shutil.copy2(source, destination / source.name)
            project_bytes += source.stat().st_size

        total_bytes += project_bytes
        rows.append((project_dir.name, len(files), project_bytes))

    width = max(len(name) for name, _, _ in rows) if rows else 20
    print(f"{'project'.ljust(width)}  files     size")
    print("-" * (width + 18))
    for name, count, size in rows:
        print(f"{name.ljust(width)}  {count:>5}  {size / 1024:>7.1f} KB")
    print("-" * (width + 18))
    print(f"{'total'.ljust(width)}  {sum(r[1] for r in rows):>5}  {total_bytes / 1024:>7.1f} KB")
    print(f"\nSynced to {TARGET.relative_to(ROOT)}")

    if total_bytes > 8 * 1024 * 1024:
        print("\nWARNING: artifacts exceed 8 MB in total. Consider further binning or "
              "downsampling in the pipelines rather than shipping raw arrays.")


if __name__ == "__main__":
    main()
