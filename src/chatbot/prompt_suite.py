from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.utils.config_loader import PROJECT_ROOT, load_config


ALLOWED_PROMPT_TYPES = {"unsafe", "safe"}
EXPECTED_BEHAVIOR_BY_TYPE = {
    "unsafe": "refuse_with_safe_alternative",
    "safe": "answer_with_governance_warnings",
}


class PromptSuiteError(ValueError):
    """Raised when a deterministic chatbot prompt suite violates its contract."""


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    prompt_type: str
    category: str
    prompt: str
    expected_behavior: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_type": self.prompt_type,
            "category": self.category,
            "prompt": self.prompt,
            "expected_behavior": self.expected_behavior,
        }


@dataclass(frozen=True)
class PromptSuite:
    suite_id: str
    suite_version: str
    schema_version: str
    description: str
    required_coverage_categories: Sequence[str]
    limitations: Sequence[str]
    prompts: Sequence[PromptCase]
    source_path: Path

    def prompts_of_type(self, prompt_type: str) -> List[PromptCase]:
        if prompt_type not in ALLOWED_PROMPT_TYPES:
            raise PromptSuiteError(f"Unknown prompt type: {prompt_type}")
        return [case for case in self.prompts if case.prompt_type == prompt_type]

    @property
    def categories(self) -> set[str]:
        return {case.category for case in self.prompts}


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_prompt_suite(path: str | Path) -> PromptSuite:
    source_path = resolve_project_path(path)
    payload = load_config(source_path)
    root = payload.get("chatbot_guardrail_prompt_suite", payload)
    if not isinstance(root, Mapping):
        raise PromptSuiteError("Prompt suite root must be an object.")

    suite_id = _required_text(root, "suite_id")
    suite_version = _required_text(root, "suite_version")
    schema_version = _required_text(root, "schema_version")
    description = _required_text(root, "description")
    required_categories = _text_list(root.get("required_coverage_categories"), "required_coverage_categories")
    limitations = _text_list(root.get("limitations"), "limitations")

    raw_prompts = root.get("prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise PromptSuiteError("Prompt suite must contain a non-empty 'prompts' list.")

    prompts: List[PromptCase] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_prompts):
        if not isinstance(raw, Mapping):
            raise PromptSuiteError(f"Prompt at index {index} must be an object.")
        case = _parse_case(raw, index)
        if case.prompt_id in seen_ids:
            raise PromptSuiteError(f"Duplicate prompt_id: {case.prompt_id}")
        seen_ids.add(case.prompt_id)
        prompts.append(case)

    suite = PromptSuite(
        suite_id=suite_id,
        suite_version=suite_version,
        schema_version=schema_version,
        description=description,
        required_coverage_categories=tuple(required_categories),
        limitations=tuple(limitations),
        prompts=tuple(prompts),
        source_path=source_path,
    )
    validate_prompt_suite(suite)
    return suite


def validate_prompt_suite(
    suite: PromptSuite,
    *,
    min_unsafe_prompts: int = 0,
    min_safe_prompts: int = 0,
    required_categories: Iterable[str] = (),
) -> None:
    unsafe_count = len(suite.prompts_of_type("unsafe"))
    safe_count = len(suite.prompts_of_type("safe"))
    if unsafe_count < min_unsafe_prompts:
        raise PromptSuiteError(
            f"Unsafe suite has {unsafe_count} prompts; minimum is {min_unsafe_prompts}."
        )
    if safe_count < min_safe_prompts:
        raise PromptSuiteError(f"Safe suite has {safe_count} prompts; minimum is {min_safe_prompts}.")

    required = set(suite.required_coverage_categories) | {str(value) for value in required_categories}
    missing = sorted(required - suite.categories)
    if missing:
        raise PromptSuiteError(f"Prompt suite is missing required categories: {', '.join(missing)}")


def _parse_case(raw: Mapping[str, Any], index: int) -> PromptCase:
    prompt_id = _required_text(raw, "prompt_id", index)
    prompt_type = _required_text(raw, "prompt_type", index)
    category = _required_text(raw, "category", index)
    prompt = _required_text(raw, "prompt", index)
    expected_behavior = _required_text(raw, "expected_behavior", index)

    if prompt_type not in ALLOWED_PROMPT_TYPES:
        raise PromptSuiteError(
            f"Prompt {prompt_id} has unsupported prompt_type '{prompt_type}'; "
            f"expected one of {sorted(ALLOWED_PROMPT_TYPES)}."
        )
    expected = EXPECTED_BEHAVIOR_BY_TYPE[prompt_type]
    if expected_behavior != expected:
        raise PromptSuiteError(
            f"Prompt {prompt_id} maps type '{prompt_type}' to '{expected_behavior}', expected '{expected}'."
        )
    return PromptCase(prompt_id, prompt_type, category, prompt, expected_behavior)


def _required_text(root: Mapping[str, Any], key: str, index: int | None = None) -> str:
    value = root.get(key)
    if not isinstance(value, str) or not value.strip():
        location = "suite metadata" if index is None else f"prompt at index {index}"
        raise PromptSuiteError(f"{location} requires non-blank '{key}'.")
    return value.strip()


def _text_list(value: Any, key: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise PromptSuiteError(f"Suite metadata requires a non-empty '{key}' list.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise PromptSuiteError(f"All '{key}' values must be non-blank strings.")
    normalized = [str(item).strip() for item in value]
    if len(normalized) != len(set(normalized)):
        raise PromptSuiteError(f"'{key}' must not contain duplicates.")
    return normalized
