#!/usr/bin/env python3
"""Package the Skyline-PRISM External Tool as a ZIP file.

This script creates a distributable ZIP file that can be installed
via Skyline's External Tools dialog.
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import datetime
from pathlib import Path


def create_package(output_dir: Path | None = None) -> Path:
    """Create the Skyline External Tool ZIP package.

    Args:
        output_dir: Directory to write the ZIP file. Defaults to project root.

    Returns:
        Path to the created ZIP file.

    """
    # Paths
    project_root = Path(__file__).parent.parent
    tool_dir = project_root / "skyline-external-tool"

    if output_dir is None:
        output_dir = project_root

    # Get version from pyproject.toml or use date
    try:
        import tomllib

        with open(project_root / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        version = pyproject["project"]["version"]
    except Exception:
        version = datetime.now().strftime("%Y%m%d")

    # Output filename
    zip_filename = f"SkylinePRISM-{version}.zip"
    zip_path = output_dir / zip_filename

    print(f"Creating {zip_path}")

    # Create ZIP file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add all files from skyline-external-tool directory
        for file_path in tool_dir.rglob("*"):
            if file_path.is_file():
                # Calculate archive name (relative to tool_dir)
                arcname = file_path.relative_to(tool_dir)
                print(f"  Adding: {arcname}")
                zf.write(file_path, arcname)

        # Add skyline_prism source package
        source_dir = project_root / "skyline_prism"
        print(f"  Bundling source package: {source_dir}")
        for file_path in source_dir.rglob("*"):
            if file_path.is_file() and "__pycache__" not in file_path.parts:
                # Add to python_prism/ inside zip
                arcname = Path("skyline_prism") / file_path.relative_to(source_dir)
                zf.write(file_path, arcname)

    print(f"\nPackage created: {zip_path}")
    print(f"  Size: {zip_path.stat().st_size:,} bytes")

    return zip_path


def main() -> int:
    """Run the packaging script."""
    import argparse

    parser = argparse.ArgumentParser(description="Package Skyline-PRISM External Tool")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for the ZIP file",
    )
    args = parser.parse_args()

    try:
        create_package(args.output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
