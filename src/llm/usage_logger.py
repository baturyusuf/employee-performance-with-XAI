from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from src.llm.cost_estimator import OPENAI_STANDARD_SHORT_CONTEXT_PRICES
from src.utils.config import SETTINGS
from src.utils.experiment_registry import utc_now_iso


USAGE_LOG_PATH = SETTINGS.reports_dir / "llm_explanations" / "llm_usage_log.csv"


def append_llm_usage(
    *,
    run_id: str,
    case_id: str,
    operation: str,
    provider: str,
    model: str,
    usage: Dict[str, Any],
    notes: str = "",
) -> Dict[str, Any]:
    row = llm_usage_row(
        run_id=run_id,
        case_id=case_id,
        operation=operation,
        provider=provider,
        model=model,
        usage=usage,
        notes=notes,
    )
    USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not USAGE_LOG_PATH.exists()
    with USAGE_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def llm_usage_row(
    *,
    run_id: str,
    case_id: str,
    operation: str,
    provider: str,
    model: str,
    usage: Dict[str, Any],
    notes: str = "",
) -> Dict[str, Any]:
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    requests = int(usage.get("requests") or 1 if total_tokens else 0)
    cached_tokens = _cached_input_tokens(usage)
    reasoning_tokens = _reasoning_tokens(usage)
    cost = estimate_actual_usage_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
    )
    return {
        "timestamp": utc_now_iso(),
        "run_id": run_id,
        "case_id": case_id,
        "operation": operation,
        "provider": provider,
        "model": model,
        "requests": requests,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
        "notes": notes,
    }


def estimate_actual_usage_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    price = OPENAI_STANDARD_SHORT_CONTEXT_PRICES.get(model)
    if price is None:
        return 0.0
    cached = min(cached_input_tokens, input_tokens)
    uncached = max(0, input_tokens - cached)
    input_cost = uncached * price.input_per_million / 1_000_000
    cached_cost = cached * (price.cached_input_per_million or price.input_per_million) / 1_000_000
    output_cost = output_tokens * price.output_per_million / 1_000_000
    return round(input_cost + cached_cost + output_cost, 8)


def normalize_usage_object(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if is_dataclass(usage):
        return asdict(usage)
    out = {}
    for key in [
        "requests",
        "input_tokens",
        "prompt_tokens",
        "output_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens_details",
        "prompt_tokens_details",
        "output_tokens_details",
        "completion_tokens_details",
    ]:
        if hasattr(usage, key):
            out[key] = getattr(usage, key)
    return out


def summarize_usage_log(path: Path = USAGE_LOG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {
            "rows": 0,
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "estimated_total_cost_usd": 0.0,
        }
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "rows": len(rows),
        "total_requests": sum(int(float(row.get("requests") or 0)) for row in rows),
        "total_input_tokens": sum(int(float(row.get("input_tokens") or 0)) for row in rows),
        "total_cached_input_tokens": sum(int(float(row.get("cached_input_tokens") or 0)) for row in rows),
        "total_output_tokens": sum(int(float(row.get("output_tokens") or 0)) for row in rows),
        "total_tokens": sum(int(float(row.get("total_tokens") or 0)) for row in rows),
        "estimated_total_cost_usd": round(
            sum(float(row.get("estimated_cost_usd") or 0.0) for row in rows),
            6,
        ),
    }


def _cached_input_tokens(usage: Dict[str, Any]) -> int:
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    if is_dataclass(details):
        details = asdict(details)
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    if isinstance(details, dict):
        return int(details.get("cached_tokens") or 0)
    return 0


def _reasoning_tokens(usage: Dict[str, Any]) -> int:
    details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    if is_dataclass(details):
        details = asdict(details)
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    if isinstance(details, dict):
        return int(details.get("reasoning_tokens") or 0)
    return 0
