from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.governance.warning_taxonomy import MANDATORY_WARNING_IDS, warning_taxonomy_rows
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


OUTPUT_DIR = SETTINGS.reports_dir / "governance_reports"


def run(output_dir: Path = OUTPUT_DIR, append_registry: bool = True) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = warning_taxonomy_rows()
    csv_path = output_dir / "warning_taxonomy.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    md_path = output_dir / "warning_taxonomy.md"
    md_path.write_text(render_markdown(rows), encoding="utf-8")
    if append_registry:
        append_registry_row(
            {
                "run_id": f"warning_taxonomy_report_{utc_now_iso()}",
                "date_time": utc_now_iso(),
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.governance.warning_taxonomy_report",
                "config": "canonical warning taxonomy for governed explanations and agents",
                "feature_set": "not_applicable",
                "model": "governance_warning_taxonomy",
                "seed": "deterministic",
                "cv_strategy": "not_applicable",
                "primary_metrics": f"warnings={len(rows)}; mandatory={len(MANDATORY_WARNING_IDS)}",
                "output_dir": "reports/governance_reports",
                "notes": "Wrote canonical warning taxonomy used for LLM explanations and agent audits.",
                "decision_status": "accepted",
            }
        )
    return {"csv": str(csv_path), "markdown": str(md_path)}


def render_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Canonical Governance Warning Taxonomy",
        "",
        "This taxonomy normalizes LLM and agent warning text into stable IDs so evaluations compare warning concepts rather than surface wording.",
        "",
        "## Mandatory Warning IDs",
        "",
    ]
    for warning_id in MANDATORY_WARNING_IDS:
        lines.append(f"- `{warning_id}`")
    lines.extend(
        [
            "",
            "## Warning Table",
            "",
            "| Warning ID | Category | Severity | Mandatory | Canonical Message |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['warning_id']}`",
                    str(row["category"]),
                    str(row["severity"]),
                    str(row["mandatory"]),
                    str(row["canonical_message"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- This is not an ethics score.",
            "- The taxonomy supports reproducible evaluation of warning coverage and consistency.",
            "- Case-specific agents may emit only the relevant subset of warnings; mandatory warnings are enforced at the governed explanation layer.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write canonical governance warning taxonomy report.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(output_dir=args.output_dir, append_registry=not args.no_registry))
