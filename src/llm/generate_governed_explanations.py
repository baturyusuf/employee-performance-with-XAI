from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.llm.evidence_schema import CompleteCaseEvidence, write_evidence_json
from src.llm.governed_explainer import GovernedExplainer
from src.llm.openai_client import OpenAIClientAPIError, OpenAIClientConfigurationError
from src.llm.runtime_config import LLMRuntimeConfig
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


OUTPUT_DIR = SETTINGS.reports_dir / "llm_explanations"


def reason_case_ids(limit: int) -> List[str]:
    root = SETTINGS.reports_dir / "xai" / "final_candidates" / "reason_code_examples"
    ids = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            ids.append(str(payload.get("sample_index")))
        except Exception:
            continue
    unique = []
    for item in ids:
        if item and item not in unique:
            unique.append(item)
    return unique[:limit]


def write_examples_markdown(rows: List[Dict[str, Any]], output_path: Path) -> None:
    lines = ["# Governed Explanation Examples", ""]
    for row in rows:
        out = row["output"]
        evidence = row["evidence"]
        lines.extend(
            [
                f"## Case {out['case_id']}",
                "",
                "### Structured Evidence Summary",
                f"- Predicted class: {evidence['prediction']['predicted_class']}",
                f"- Confidence: {evidence['prediction']['confidence']}",
                f"- Feature policy: {evidence['prediction']['feature_policy']}",
                "",
                "### Governed Explanation",
                out["detailed_explanation"],
                "",
                "### Compliance Check",
                f"- Pass: {out['faithfulness_check']['faithfulness_pass']}",
                f"- Score: {out['faithfulness_check']['score']}",
                f"- Missing warnings: {out['faithfulness_check']['missing_warnings']}",
                "",
                "### Warnings",
            ]
        )
        for warning in out.get("warnings", []):
            lines.append(f"- {warning['type']} / {warning['severity']}: {warning['message']}")
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    limit: int = 5,
    provider: str | None = None,
    model: str | None = None,
    require_real_llm: bool = False,
) -> Dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = LLMRuntimeConfig.from_env()
    runtime_config = LLMRuntimeConfig(
        provider=(provider or base_config.provider),  # type: ignore[arg-type]
        model=model or base_config.model,
        temperature=base_config.temperature,
        max_tokens=base_config.max_tokens,
        require_real_llm=require_real_llm or base_config.require_real_llm,
    )
    explainer = GovernedExplainer(runtime_config=runtime_config)
    rows = []
    for case_id in reason_case_ids(limit):
        evidence = CompleteCaseEvidence.from_reports(case_id=case_id)
        output = explainer.generate(evidence)
        case_dir = OUTPUT_DIR / f"case_{case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "governed_explanation.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        write_evidence_json(case_dir / "structured_evidence.json", case_id=case_id)
        rows.append({"case_id": case_id, "evidence": evidence.to_dict(), "output": output})

    examples_path = OUTPUT_DIR / "governed_explanation_examples.md"
    write_examples_markdown(rows, examples_path)
    eval_rows = []
    for row in rows:
        check = row["output"]["faithfulness_check"]
        eval_rows.append(
            {
                "case_id": row["case_id"],
                "faithfulness_pass": check["faithfulness_pass"],
                "score": check["score"],
                "unsupported_claim_count": len(check["unsupported_claims"]),
                "forbidden_claim_count": len(check["forbidden_claims"]),
                "missing_warning_count": len(check["missing_warnings"]),
            }
        )
    pd.DataFrame(eval_rows).to_csv(OUTPUT_DIR / "governed_explanation_eval.csv", index=False)
    append_registry_row(
        {
            "run_id": f"llm_governed_explanations_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.llm.generate_governed_explanations",
            "config": "structured evidence loaded from final candidate reports",
            "feature_set": "no_salary_hike_no_attrition_no_department",
            "model": f"{runtime_config.provider}:{runtime_config.model}",
            "seed": "deterministic",
            "cv_strategy": "not_applicable",
            "primary_metrics": "faithfulness score; warning coverage",
            "output_dir": "reports/llm_explanations",
            "notes": "Generated governed explanations from structured XAI evidence.",
            "decision_status": "candidate",
        }
    )
    return {"examples": examples_path, "eval": OUTPUT_DIR / "governed_explanation_eval.csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate governed explanations from structured XAI evidence.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--provider", choices=["auto", "offline", "openai"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--require-real-llm",
        action="store_true",
        help="Fail instead of falling back to the offline stub when a real LLM is unavailable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        print(run(limit=args.limit, provider=args.provider, model=args.model, require_real_llm=args.require_real_llm))
    except (OpenAIClientConfigurationError, OpenAIClientAPIError) as exc:
        raise SystemExit(f"LLM configuration error: {exc}") from exc
