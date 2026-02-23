#!/usr/bin/env python3
"""CLI entry point for the modular report system.

Usage:
    python -m report.cli --config configs/6view_comparison.yaml \\
        --output reports/6view_report.html

    # Or directly:
    python report/cli.py --config report/configs/6view_comparison.yaml \\
        --output reports/6view_report.html
"""

import argparse
import sys
from pathlib import Path

# Support running as script or module
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from report.schema import ReportConfig
from report.builder import build_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment comparison HTML report")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: reports/<config_name>.html)")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip qualitative comparison (faster)")
    args = parser.parse_args()

    config = ReportConfig.from_yaml(args.config)

    if args.no_images:
        config.visualization["sample_frames"] = []

    output = args.output
    if output is None:
        name = Path(args.config).stem
        output = f"reports/{name}.html"

    build_report(config, output)


if __name__ == "__main__":
    main()
