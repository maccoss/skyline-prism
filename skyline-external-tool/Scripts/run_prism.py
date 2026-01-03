#!/usr/bin/env python3
"""Skyline External Tool launcher for PRISM.

This script is invoked by Skyline when the user runs the PRISM External Tool.
It:
1. Ensures Python dependencies are installed.
2. Parses arguments from Skyline.
3. Launches the Unified PRISM GUI (Configuration -> Console -> Results).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add tool root to sys.path to find bundled skyline_prism package
# The script is in Scripts/run_prism.py, so root is two levels up
tool_root = Path(__file__).parent.parent
if str(tool_root) not in sys.path:
    sys.path.insert(0, str(tool_root))


def ensure_dependencies() -> bool:
    """Check and install required Python dependencies."""
    required_packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pyarrow",
        "pyyaml",
        "duckdb",
        "PyQt6",
        "PyQt6-WebEngine",
        "matplotlib",
        "seaborn",
    ]

    import importlib.util
    import subprocess

    missing = []
    package_map = {
        "pyyaml": "yaml",
        "scikit-learn": "sklearn",
        "PyQt6-WebEngine": "PyQt6.QtWebEngineWidgets",
    }

    print("Checking dependencies...")
    for pkg in required_packages:
        module_name = package_map.get(pkg, pkg)
        if "." in module_name:
            base, _ = module_name.split(".", 1)
            if importlib.util.find_spec(base) is None:
                missing.append(pkg)
                continue
        else:
            if importlib.util.find_spec(module_name) is None:
                missing.append(pkg)

    if not missing:
        return True

    print(f"Installing missing dependencies: {', '.join(missing)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error installing dependencies: {e}", file=sys.stderr)
        return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments from Skyline."""
    parser = argparse.ArgumentParser(description="Run PRISM analysis from Skyline External Tool")
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to the Skyline-PRISM report CSV file",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        nargs="+",
        required=True,
        help="Path to the Replicates report CSV file(s) (sample metadata)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for PRISM results",
    )
    return parser.parse_args()


def check_batch_column(metadata_paths: list[Path]) -> bool:
    """Check if any of the metadata files contain a 'Batch' column."""
    import pandas as pd

    for path in metadata_paths:
        try:
            # Read just the header to check columns
            df = pd.read_csv(path, nrows=0)
            columns_lower = [c.lower() for c in df.columns]
            if "batch" in columns_lower:
                return True
        except Exception:
            pass
    return False


def main() -> int:
    """Run the Skyline External Tool main entry point."""
    # Ensure dependencies are installed first
    if not ensure_dependencies():
        print("Failed to ensure dependencies. Exiting.", file=sys.stderr)
        return 1

    args = parse_args()

    # Validate input files exist
    if not args.report.exists():
        print(f"Error: Report file not found: {args.report}", file=sys.stderr)
        return 1

    for m_path in args.metadata:
        if not m_path.exists():
            print(f"Error: Metadata file not found: {m_path}", file=sys.stderr)
            return 1

    print("PRISM External Tool - Launching Unified GUI...")
    print(f"  Report: {args.report}")
    print(f"  Metadata: {args.metadata}")
    print(f"  Output: {args.output}")

    # Check for batch column
    has_batch = check_batch_column(args.metadata)

    try:
        from PyQt6.QtWidgets import QApplication

        from skyline_prism.gui.main_window import PRISMMainWindow

        app = QApplication(sys.argv)

        # Pass input files to the unified window
        # Join multiple metadata files with semicolon
        metadata_str = ";".join(str(m) for m in args.metadata)

        window = PRISMMainWindow(
            report_file=str(args.report),
            metadata_file=metadata_str,
            output_dir=str(args.output),
            has_batch_column=has_batch,
        )
        window.show()

        return app.exec()

    except ImportError as e:
        print(f"Failed to import GUI components: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error launching application: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
