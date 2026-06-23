from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from src.utils.config import SETTINGS


@dataclass(frozen=True)
class TokenPrice:
    input_per_million: float
    cached_input_per_million: float | None
    output_per_million: float


OPENAI_STANDARD_SHORT_CONTEXT_PRICES: Dict[str, TokenPrice] = {
    "gpt-5.5": TokenPrice(5.00, 0.50, 30.00),
    "gpt-5.4": TokenPrice(2.50, 0.25, 15.00),
    "gpt-5.4-mini": TokenPrice(0.75, 0.075, 4.50),
    "gpt-5.4-nano": TokenPrice(0.20, 0.02, 1.25),
    "gpt-5": TokenPrice(1.25, 0.125, 10.00),
}


def estimate_cost(
    model: str,
    n_cases: int,
    calls_per_case: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    cached_input_fraction: float = 0.0,
) -> Dict[str, float | str | int]:
    if model not in OPENAI_STANDARD_SHORT_CONTEXT_PRICES:
        raise ValueError(
            f"Unknown model price for {model}. Known models: {sorted(OPENAI_STANDARD_SHORT_CONTEXT_PRICES)}"
        )
    if not 0.0 <= cached_input_fraction <= 1.0:
        raise ValueError("cached_input_fraction must be between 0 and 1")
    price = OPENAI_STANDARD_SHORT_CONTEXT_PRICES[model]
    n_calls = n_cases * calls_per_case
    uncached_input_tokens = avg_input_tokens * n_calls * (1.0 - cached_input_fraction)
    cached_input_tokens = avg_input_tokens * n_calls * cached_input_fraction
    output_tokens = avg_output_tokens * n_calls
    input_cost = uncached_input_tokens * price.input_per_million / 1_000_000
    cached_cost = 0.0
    if price.cached_input_per_million is not None:
        cached_cost = cached_input_tokens * price.cached_input_per_million / 1_000_000
    else:
        input_cost += cached_input_tokens * price.input_per_million / 1_000_000
    output_cost = output_tokens * price.output_per_million / 1_000_000
    total = input_cost + cached_cost + output_cost
    return {
        "model": model,
        "n_cases": n_cases,
        "calls_per_case": calls_per_case,
        "total_calls": n_calls,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "cached_input_fraction": cached_input_fraction,
        "estimated_input_cost_usd": round(input_cost, 6),
        "estimated_cached_input_cost_usd": round(cached_cost, 6),
        "estimated_output_cost_usd": round(output_cost, 6),
        "estimated_total_cost_usd": round(total, 6),
    }


def write_markdown_estimate(output_path: Path) -> None:
    scenarios = [
        ("small_eval", 5, 8, 8_000, 600),
        ("paper_eval", 50, 8, 8_000, 600),
        ("dashboard_demo", 100, 8, 8_000, 600),
    ]
    models = ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5"]
    lines = [
        "# LLM Cost Estimate",
        "",
        "Prices are approximate and based on OpenAI standard short-context text token pricing per 1M tokens at the time this file was generated.",
        "Assumption: one governed explanation call, six specialist agent calls, and one supervisor call per case.",
        "",
        "| scenario | model | cases | calls | avg input/call | avg output/call | estimated USD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario_name, cases, calls_per_case, input_tokens, output_tokens in scenarios:
        for model in models:
            estimate = estimate_cost(
                model=model,
                n_cases=cases,
                calls_per_case=calls_per_case,
                avg_input_tokens=input_tokens,
                avg_output_tokens=output_tokens,
            )
            lines.append(
                f"| {scenario_name} | {model} | {cases} | {estimate['total_calls']} | "
                f"{input_tokens} | {output_tokens} | ${estimate['estimated_total_cost_usd']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Cost controls:",
            "- Start with `gpt-5.4-mini` for development and guardrail evaluation.",
            "- Use `gpt-5.5` only for final high-quality examples or manuscript artifacts.",
            "- Keep `--limit` small during iteration.",
            "- Do not run full case sweeps until token logging is reviewed.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate OpenAI LLM-agent governance cost.")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--n-cases", type=int, default=5)
    parser.add_argument("--calls-per-case", type=int, default=8)
    parser.add_argument("--avg-input-tokens", type=int, default=8000)
    parser.add_argument("--avg-output-tokens", type=int, default=600)
    parser.add_argument("--cached-input-fraction", type=float, default=0.0)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.write_report:
        path = SETTINGS.reports_dir / "llm_explanations" / "llm_cost_estimate.md"
        write_markdown_estimate(path)
        print({"report": str(path)})
    else:
        print(
            estimate_cost(
                model=args.model,
                n_cases=args.n_cases,
                calls_per_case=args.calls_per_case,
                avg_input_tokens=args.avg_input_tokens,
                avg_output_tokens=args.avg_output_tokens,
                cached_input_fraction=args.cached_input_fraction,
            )
        )
