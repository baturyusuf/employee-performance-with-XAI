from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
CLAIM_DIGEST = "1664b188df14135d3b3de8642de3d080c25a3fbadc440615c2bd04b2f14ddabe"
BASE_COMMIT = "af4668aff4f52abcb4b6294f54434f00b59123d5"
CLAIM_DIR = ROOT / "reports/research_log/major_revision_v3/phase5a_claim_matrix"
PHASE_DIR = ROOT / "reports/research_log/major_revision_v3/phase5b_manuscript"
MANUSCRIPT_DIR = ROOT / "manuscript/mdpi_information"
ASSET_DIR = MANUSCRIPT_DIR / "assets"
LITERATURE_CONFIG = ROOT / "configs/literature_positioning_v3.json"


class ManuscriptRevisionError(RuntimeError):
    """Raised when the Phase 5B evidence contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_approved_claims() -> list[dict[str, str]]:
    approval = json.loads((CLAIM_DIR / "approval_record.json").read_text(encoding="utf-8"))
    observed = approval.get("claim_set_sha256")
    if observed != CLAIM_DIGEST or approval.get("decision") != "approved":
        raise ManuscriptRevisionError(
            f"Phase 5A approval mismatch: decision={approval.get('decision')!r}, digest={observed!r}"
        )
    with (CLAIM_DIR / "CLAIM_MATRIX.csv").open(encoding="utf-8-sig", newline="") as stream:
        claims = list(csv.DictReader(stream))
    if not claims or any(row["approval_status"] != "approved" for row in claims):
        raise ManuscriptRevisionError("Every Phase 5A claim must remain approved.")
    return claims


def _escape_tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _inline_tex(value: str) -> str:
    citations: list[str] = []

    def stash_citation(match: re.Match[str]) -> str:
        keys = re.findall(r"@([A-Za-z0-9_:-]+)", match.group(0))
        citations.append(r"\cite{" + ",".join(keys) + "}")
        return f"@@CITATION{len(citations) - 1}@@"

    value = re.sub(r"\[(?:@[^\]]+)\]", stash_citation, value)
    code: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        code.append(r"\texttt{" + _escape_tex(match.group(1)) + "}")
        return f"@@CODE{len(code) - 1}@@"

    value = re.sub(r"`([^`]+)`", stash_code, value)
    value = _escape_tex(value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*", r"\\emph{\1}", value)
    for index, item in enumerate(citations):
        value = value.replace(_escape_tex(f"@@CITATION{index}@@"), item)
    for index, item in enumerate(code):
        value = value.replace(_escape_tex(f"@@CODE{index}@@"), item)
    return value


def _split_markdown_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    cursor = start
    while cursor < len(lines) and lines[cursor].strip().startswith("|"):
        cells = [cell.strip() for cell in lines[cursor].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            rows.append(cells)
        cursor += 1
    return rows, cursor


def markdown_to_tex(markdown: str) -> str:
    title_match = re.search(r"^# (.+)$", markdown, flags=re.MULTILINE)
    if not title_match:
        raise ManuscriptRevisionError("The Markdown manuscript has no title.")
    title = title_match.group(1).strip()
    abstract_match = re.search(
        r"^## Abstract\s*$\n(.+?)(?=\n\*\*Keywords:\*\*)", markdown, flags=re.MULTILINE | re.DOTALL
    )
    keywords_match = re.search(r"^\*\*Keywords:\*\* (.+)$", markdown, flags=re.MULTILINE)
    if not abstract_match or not keywords_match:
        raise ManuscriptRevisionError("Abstract or keywords are missing.")

    preamble = rf"""% Generated from main.md by src/governance/manuscript_revision_v3.py.
% Place this content in the current official MDPI Information template.
\documentclass[information,article,submit,pdftex,moreauthors]{{Definitions/mdpi}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{graphicx}}
\usepackage{{float}}
\firstpage{{1}}
\makeatletter\setcounter{{page}}{{\@firstpage}}\makeatother
\pubvolume{{AUTHOR\_TO\_COMPLETE}}
\issuenum{{AUTHOR\_TO\_COMPLETE}}
\articlenumber{{AUTHOR\_TO\_COMPLETE}}
\pubyear{{2026}}
\copyrightyear{{2026}}
\datereceived{{AUTHOR\_TO\_COMPLETE}}
\daterevised{{AUTHOR\_TO\_COMPLETE}}
\dateaccepted{{AUTHOR\_TO\_COMPLETE}}
\datepublished{{AUTHOR\_TO\_COMPLETE}}
\hreflink{{https://doi.org/AUTHOR\_TO\_COMPLETE}}
\Title{{{_inline_tex(title)}}}
\TitleCitation{{Leakage- and Governance-Aware XAI Audit Protocol}}
\newcommand{{\orcidauthorA}}{{AUTHOR\_TO\_COMPLETE}}
\newcommand{{\orcidauthorB}}{{AUTHOR\_TO\_COMPLETE}}
\Author{{Muhammed Yusuf Batur $^{{1,*}}$\orcidA{{}} and Mehmet G{{\"o}}kt{{\"u}}rk $^{{2}}$\orcidB{{}}}}
\AuthorNames{{Muhammed Yusuf Batur and Mehmet G{{\"o}}kt{{\"u}}rk}}
\AuthorCitation{{Batur, M.Y.; G{{\"o}}kt{{\"u}}rk, M.}}
\address{{%
$^{{1}}$ \quad Rumeli University, AUTHOR\_TO\_COMPLETE; myusuf.batur@rumeli.edu.tr\\
$^{{2}}$ \quad Gebze Technical University, AUTHOR\_TO\_COMPLETE; gokturk@gtu.edu.tr}}
\corres{{Correspondence: myusuf.batur@rumeli.edu.tr; AUTHOR\_TO\_COMPLETE}}
\abstract{{{_inline_tex(' '.join(abstract_match.group(1).split()))}}}
\keyword{{{_inline_tex(keywords_match.group(1))}}}
\externalbibliography{{yes}}
\begin{{document}}
"""

    body_start = re.search(r"^## 1\. ", markdown, flags=re.MULTILINE)
    if not body_start:
        raise ManuscriptRevisionError("Numbered manuscript body is missing.")
    body = markdown[body_start.start() :]
    lines = body.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    list_open = False
    pending_table_caption: str | None = None

    declaration_commands = {
        "Author Contributions": "authorcontributions",
        "Funding": "funding",
        "Institutional Review Board Statement": "institutionalreview",
        "Informed Consent Statement": "informedconsent",
        "Data Availability Statement": "dataavailability",
        "Conflicts of Interest": "conflictsofinterest",
    }

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            output.append(_inline_tex(" ".join(item.strip() for item in paragraph)) + "\n")
            paragraph = []

    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        stripped = line.strip()
        if stripped.startswith("<!--"):
            cursor += 1
            continue
        if not stripped:
            flush_paragraph()
            if list_open:
                output.append("\\end{itemize}\n")
                list_open = False
            cursor += 1
            continue
        heading = re.match(r"^(#{2,3}) (?:\d+(?:\.\d+)*\.? )?(.+)$", stripped)
        if heading:
            flush_paragraph()
            if list_open:
                output.append("\\end{itemize}\n")
                list_open = False
            level, text = heading.groups()
            if text == "References":
                cursor += 1
                continue
            if text in declaration_commands:
                cursor += 1
                declaration_lines: list[str] = []
                while cursor < len(lines) and not lines[cursor].startswith("## "):
                    if lines[cursor].strip() and not lines[cursor].strip().startswith("<!--"):
                        declaration_lines.append(lines[cursor].strip())
                    cursor += 1
                output.append(
                    f"\\{declaration_commands[text]}{{{_inline_tex(' '.join(declaration_lines))}}}\n"
                )
                continue
            if text == "Acknowledgments":
                cursor += 1
                declaration_lines = []
                while cursor < len(lines) and not lines[cursor].startswith("## "):
                    if lines[cursor].strip():
                        declaration_lines.append(lines[cursor].strip())
                    cursor += 1
                output.append(f"\\acknowledgments{{{_inline_tex(' '.join(declaration_lines))}}}\n")
                continue
            if text == "Use of AI-Assisted Technologies":
                output.append("\\section*{Use of AI-Assisted Technologies}\n")
            else:
                command = "section" if level == "##" else "subsection"
                output.append(f"\\{command}{{{_inline_tex(text)}}}\n")
            cursor += 1
            continue
        table_caption = re.match(r"^\*\*Table (\d+)\. (.+)\*\*$", stripped)
        if table_caption:
            flush_paragraph()
            pending_table_caption = table_caption.group(2)
            cursor += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            rows, cursor = _split_markdown_table(lines, cursor)
            if not rows:
                continue
            width = len(rows[0])
            label_number = len([item for item in output if "\\begin{table}" in item]) + 1
            tex_rows = [" & ".join(_inline_tex(cell) for cell in row) + r" \\" for row in rows]
            table = [
                "\\begin{table}[H]",
                "\\centering\\scriptsize",
                f"\\caption{{{_inline_tex(pending_table_caption or 'Results table')}}}",
                f"\\label{{tab:{label_number}}}",
                "\\resizebox{\\textwidth}{!}{%",
                "\\begin{tabular}{" + "l" * width + "}",
                "\\toprule",
                tex_rows[0],
                "\\midrule",
                *tex_rows[1:],
                "\\bottomrule",
                "\\end{tabular}}",
                "\\end{table}\n",
            ]
            output.append("\n".join(table))
            pending_table_caption = None
            continue
        figure = re.match(r"^!\[(Figure (\d+)\. .+)\]\(([^)]+)\)$", stripped)
        if figure:
            flush_paragraph()
            caption, number, path = figure.groups()
            output.append(
                "\n".join(
                    [
                        "\\begin{figure}[H]",
                        "\\centering",
                        f"\\includegraphics[width=0.98\\textwidth]{{\\detokenize{{{path}}}}}",
                        f"\\caption{{{_inline_tex(caption.split('. ', 1)[1])}}}",
                        f"\\label{{fig:{number}}}",
                        "\\end{figure}\n",
                    ]
                )
            )
            cursor += 1
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            if not list_open:
                output.append("\\begin{itemize}")
                list_open = True
            output.append(f"\\item {_inline_tex(stripped[2:])}")
            cursor += 1
            continue
        paragraph.append(stripped)
        cursor += 1
    flush_paragraph()
    if list_open:
        output.append("\\end{itemize}\n")
    return preamble + "\n".join(output) + "\n\\bibliography{references}\n\\end{document}\n"


def write_verified_bibliography(path: Path) -> None:
    config = json.loads(LITERATURE_CONFIG.read_text(encoding="utf-8"))
    blocks: list[str] = []
    for study in config["studies"]:
        authors = " and ".join(item.strip() for item in study["authors"].split(";"))
        fields = [
            f"  author = {{{_escape_tex(authors)}}}",
            f"  title = {{{_escape_tex(study['title'])}}}",
            f"  year = {{{study['year']}}}",
            f"  howpublished = {{{_escape_tex(study['venue'])}}}",
            f"  url = {{{study['primary_url']}}}",
        ]
        if study.get("doi") and study["doi_status"] == "registered":
            fields.append(f"  doi = {{{study['doi']}}}")
        blocks.append(f"@misc{{{study['citation_key']},\n" + ",\n".join(fields) + "\n}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8", newline="\n")


def validate_manuscript(markdown: str, tex: str, claims: list[dict[str, str]]) -> dict[str, object]:
    required = [
        "## 1. Introduction",
        "## 2. Related Work",
        "## 3. Materials and Methods",
        "## 4. Results",
        "## 5. Discussion",
        "## 6. Limitations",
        "## 7. Conclusions",
        "## Author Contributions",
        "## Funding",
        "## Institutional Review Board Statement",
        "## Informed Consent Statement",
        "## Data Availability Statement",
        "## Conflicts of Interest",
    ]
    missing = [heading for heading in required if heading not in markdown]
    if missing:
        raise ManuscriptRevisionError(f"Missing manuscript sections: {missing}")
    obsolete = ["multi-agent", "chatbot", "counterfactual", "gpt-5.4-mini", "v0.3-real-llm"]
    found_obsolete = [term for term in obsolete if term.lower() in markdown.lower()]
    if found_obsolete:
        raise ManuscriptRevisionError(f"Obsolete v1 concepts remain: {found_obsolete}")
    prohibited = [
        "leakage-free",
        "world first",
        "first ever",
        "deployment-ready",
        "proves fairness",
        "proves discrimination",
    ]
    found_prohibited = [term for term in prohibited if term.lower() in markdown.lower()]
    if found_prohibited:
        raise ManuscriptRevisionError(f"Prohibited wording remains: {found_prohibited}")
    cited = {
        key
        for citation in re.findall(r"\[(?:@[^\]]+)\]", markdown)
        for key in re.findall(r"@([A-Za-z0-9_:-]+)", citation)
    }
    literature = json.loads(LITERATURE_CONFIG.read_text(encoding="utf-8"))
    verified = {study["citation_key"] for study in literature["studies"]}
    unverified = sorted(cited - verified)
    if unverified:
        raise ManuscriptRevisionError(f"Unverified citation keys: {unverified}")
    claim_ids = set(re.findall(r"C\d{3}", markdown))
    expected_numeric = {row["claim_id"] for row in claims if row["claim_type"] == "numerical"}
    missing_claims = sorted(expected_numeric - claim_ids)
    if missing_claims:
        raise ManuscriptRevisionError(f"Numerical claims not traced in manuscript: {missing_claims}")
    if markdown.count("![Figure ") != 7 or markdown.count("**Table ") < 7:
        raise ManuscriptRevisionError("The manuscript must contain seven figures and at least seven tables.")
    if tex.count("\\begin{figure}") != 7 or tex.count("\\begin{table}") < 7:
        raise ManuscriptRevisionError("Markdown-to-LaTeX figure/table parity failed.")
    return {
        "claim_digest": CLAIM_DIGEST,
        "approved_claim_count": len(claims),
        "numeric_claim_ids_traced": len(expected_numeric),
        "verified_citation_count_used": len(cited),
        "figure_count": markdown.count("![Figure "),
        "table_count": markdown.count("**Table "),
        "obsolete_concept_count": 0,
        "prohibited_wording_count": 0,
        "markdown_tex_figure_parity": True,
        "markdown_tex_table_parity": True,
    }


def _write_final_tables(markdown: str, claims: list[dict[str, str]]) -> None:
    destination = PHASE_DIR / "FINAL_TABLES"
    destination.mkdir(parents=True, exist_ok=True)
    lines = markdown.splitlines()
    claim_map = {
        1: ("C001", "C703", "C801", "C802", "C803", "C804"),
        2: ("C001",),
        3: ("C101", "C102", "C103", "C104", "C105", "C106"),
        4: ("C201", "C202", "C203", "C204", "C205"),
        5: ("C301", "C302", "C303"),
        6: ("C401", "C402", "C403", "C501", "C502"),
        7: ("C601", "C602", "C603", "C604", "C605", "C606"),
        8: ("C701", "C702", "C703", "C802", "C803", "C804"),
    }
    by_id = {row["claim_id"]: row for row in claims}
    cursor = 0
    found = 0
    while cursor < len(lines):
        match = re.match(r"^\*\*Table (\d+)\. (.+)\*\*$", lines[cursor].strip())
        if not match:
            cursor += 1
            continue
        number = int(match.group(1))
        caption = match.group(2)
        cursor += 1
        while cursor < len(lines) and not lines[cursor].strip().startswith("|"):
            cursor += 1
        rows, cursor = _split_markdown_table(lines, cursor)
        if not rows:
            raise ManuscriptRevisionError(f"Table {number} has no rows.")
        stem = f"table_{number:02d}"
        with (destination / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(rows)
        preview = [
            f"# Table {number}. {caption}",
            "",
            "| " + " | ".join(rows[0]) + " |",
            "| " + " | ".join("---" for _ in rows[0]) + " |",
        ]
        preview.extend("| " + " | ".join(row) + " |" for row in rows[1:])
        (destination / f"{stem}.md").write_text("\n".join(preview) + "\n", encoding="utf-8")
        mapped = [by_id[claim_id] for claim_id in claim_map[number]]
        source_map = {
            "table_number": number,
            "caption": caption,
            "claim_set_sha256": CLAIM_DIGEST,
            "claim_ids": list(claim_map[number]),
            "sources": [
                {
                    "claim_id": row["claim_id"],
                    "source_path": row["source_path"],
                    "source_sha256": row["source_sha256"],
                    "source_selector": row["source_selector"],
                }
                for row in mapped
            ],
        }
        (destination / f"{stem}.source_map.json").write_text(
            json.dumps(source_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        found += 1
    if found != 8:
        raise ManuscriptRevisionError(f"Expected eight final tables, found {found}.")


def _write_final_figures(markdown: str) -> None:
    destination = PHASE_DIR / "FINAL_FIGURES"
    destination.mkdir(parents=True, exist_ok=True)
    figures = re.findall(r"^!\[(Figure (\d+)\. (.+))\]\(([^)]+)\)$", markdown, flags=re.MULTILINE)
    if len(figures) != 7:
        raise ManuscriptRevisionError(f"Expected seven final figures, found {len(figures)}.")
    for full_caption, number, caption, relative_source in figures:
        source_png = MANUSCRIPT_DIR / relative_source
        source_svg = source_png.with_suffix(".svg")
        if not source_png.is_file() or not source_svg.is_file():
            raise ManuscriptRevisionError(f"Missing canonical figure source: {source_png}")
        clean_stem = re.sub(r"^figure_\d+_", "", source_png.stem)
        target_stem = f"figure_{int(number):02d}_{clean_stem}"
        shutil.copy2(source_png, destination / f"{target_stem}.png")
        shutil.copy2(source_svg, destination / f"{target_stem}.svg")
        (destination / f"{target_stem}.caption.md").write_text(
            f"# Figure {number}\n\n{caption}\n", encoding="utf-8"
        )
        source_number = re.match(r"^figure_(\d+)_", source_png.stem)
        canonical_alt = ASSET_DIR / "figures/alt_text" / f"figure_{source_number.group(1)}_alt_text.txt"
        if canonical_alt.is_file():
            shutil.copy2(canonical_alt, destination / f"{target_stem}.alt.txt")


def _write_results_comparison(path: Path, claims: list[dict[str, str]]) -> None:
    fields = [
        "claim_id",
        "component",
        "proposed_claim",
        "exact_value",
        "display_value",
        "evidence_scope",
        "mandatory_qualifier",
        "prohibited_overclaim",
        "source_path",
        "source_sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fields}
            for row in claims
            if row["claim_type"] == "numerical"
        )


def _write_diff(path: Path) -> None:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--unified=3",
            BASE_COMMIT,
            "--",
            "manuscript/mdpi_information/main.md",
            "manuscript/mdpi_information/main.tex",
            "manuscript/mdpi_information/references.bib",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    path.write_text(result.stdout, encoding="utf-8", newline="\n")


BUILDER_OUTPUTS = (
    "FINAL_TABLES",
    "FINAL_FIGURES",
    "RESULTS_COMPARISON.csv",
    "ORIGINAL_VS_REVISED.diff",
    "MANUSCRIPT_VALIDATION.json",
    "manifest.json",
)


def _remove_builder_outputs() -> None:
    phase_root = PHASE_DIR.resolve()
    allowed_parent = (ROOT / "reports/research_log/major_revision_v3").resolve()
    if phase_root.parent != allowed_parent:
        raise ManuscriptRevisionError(f"Refusing unsafe output cleanup: {phase_root}")
    for name in BUILDER_OUTPUTS:
        target = (PHASE_DIR / name).resolve()
        if target.parent != phase_root:
            raise ManuscriptRevisionError(f"Refusing unsafe builder target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def build(*, replace: bool = False) -> dict[str, object]:
    claims = load_approved_claims()
    markdown_path = MANUSCRIPT_DIR / "main.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    tex = markdown_to_tex(markdown)

    write_verified_bibliography(MANUSCRIPT_DIR / "references.bib")
    (MANUSCRIPT_DIR / "main.tex").write_text(tex, encoding="utf-8", newline="\n")
    validation = validate_manuscript(markdown, tex, claims)

    generated_exist = any((PHASE_DIR / name).exists() for name in BUILDER_OUTPUTS)
    if generated_exist and not replace:
        raise ManuscriptRevisionError("Phase 5B generated output exists; pass --replace-output.")
    if replace:
        _remove_builder_outputs()

    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    _write_final_tables(markdown, claims)
    _write_final_figures(markdown)
    _write_results_comparison(PHASE_DIR / "RESULTS_COMPARISON.csv", claims)
    _write_diff(PHASE_DIR / "ORIGINAL_VS_REVISED.diff")
    (PHASE_DIR / "MANUSCRIPT_VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validation


def finalize_manifest() -> dict[str, object]:
    excluded = {"manifest.json"}
    files = [path for path in PHASE_DIR.rglob("*") if path.is_file() and path.name not in excluded]
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    payload = {
        "schema_version": 1,
        "phase": "5B",
        "claim_set_sha256": CLAIM_DIGEST,
        "base_commit": BASE_COMMIT,
        "paid_api_calls": 0,
        "raw_employee_rows_published": 0,
        "file_count": len(records),
        "files": records,
    }
    (PHASE_DIR / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the approved Phase 5B manuscript package.")
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--finalize-manifest", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build(replace=args.replace_output)
    if args.finalize_manifest:
        result["manifest"] = finalize_manifest()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
