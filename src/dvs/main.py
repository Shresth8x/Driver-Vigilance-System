from __future__ import annotations

from pathlib import Path
from typing import Optional
import argparse
import logging
import sys

from .config import apply_cli_overrides, load_config


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Driver Vigilance Monitoring System",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "config" / "default_config.json"),
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index. Use 0 for default webcam, 1 for another connected camera.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Simulated cabin temperature in Celsius.",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable audible alerts.",
    )
    parser.add_argument(
        "--hide-landmarks",
        action="store_true",
        help="Start with eye landmark points hidden.",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="Optional path for saving an MP4 demo recording.",
    )
    return parser.parse_args(argv)


def setup_logging(log_dir: str, level: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "driver_vigilance.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    config = apply_cli_overrides(
        config,
        camera_index=args.camera,
        no_sound=args.no_sound,
        show_landmarks=False if args.hide_landmarks else None,
    )
    setup_logging(config.logging.log_dir, config.logging.level)

    from .app import DriverMonitoringApp

    app = DriverMonitoringApp(
        config,
        temperature_celsius=args.temperature,
        record_path=args.record,
    )
    return app.run()
