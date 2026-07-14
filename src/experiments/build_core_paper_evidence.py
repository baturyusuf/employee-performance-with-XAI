"""Fail-closed entry point for the canonical core paper evidence scope."""

from __future__ import annotations

import argparse
import json

from src.experiments.build_manuscript_evidence import build
from src.governance.manuscript_contract import DEFAULT_CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the core leakage-aware XAI evidence package.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-reuse-compatible", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = build(
        arguments.config,
        run_id=arguments.run_id,
        reuse_compatible=not arguments.no_reuse_compatible,
        evidence_scope="core",
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, sort_keys=True))
