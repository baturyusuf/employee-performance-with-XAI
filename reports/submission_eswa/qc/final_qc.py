"""Deterministic pre-submission checks for the ESWA author-review package."""
from pathlib import Path
from hashlib import sha256
import json, re, zipfile

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "reports/submission_eswa"
MAIN = (BASE / "manuscript/main.md").read_text(encoding="utf-8-sig")
BIB = (BASE / "manuscript/references.bib").read_text(encoding="utf-8-sig")
HIGHLIGHTS = (BASE / "highlights/HIGHLIGHTS.md").read_text(encoding="utf-8-sig")
TITLE = "Auditing Intelligent Employee-Performance Prediction Systems Beyond Accuracy: A Traceable XAI Protocol"

checks = []
def check(name, passed, evidence, severity="required"):
    checks.append({"check":name,"passed":bool(passed),"severity":severity,"evidence":str(evidence)})

abstract = re.search(r"## Abstract\s+(.*?)\s+\*\*Keywords:", MAIN, re.S).group(1)
abstract_words = re.findall(r"\b[\w’-]+\b", abstract)
check("Abstract length", len(abstract_words) <= 250, f"{len(abstract_words)} words; limit 250")
keywords = [x.strip() for x in re.search(r"\*\*Keywords:\*\*\s*(.+)", MAIN).group(1).split(";")]
check("Keyword count", 1 <= len(keywords) <= 7, f"{len(keywords)} keywords; journal range 1–7; project cap 6")

highlight_lines = [line[2:].strip() for line in HIGHLIGHTS.splitlines() if line.startswith("- ")]
check("Highlight count", 3 <= len(highlight_lines) <= 5, f"{len(highlight_lines)} highlights")
for index, value in enumerate(highlight_lines, 1):
    check(f"Highlight {index} length", len(value) <= 85, f"{len(value)} characters; limit 85")

bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", BIB))
cited_keys = set(re.findall(r"(?<!\w)@([A-Za-z0-9_:-]+)", MAIN))
check("Citation closure: cited keys resolve", not (cited_keys-bib_keys), f"unresolved={sorted(cited_keys-bib_keys)}")
check("Citation closure: bibliography entries cited", not (bib_keys-cited_keys), f"uncited={sorted(bib_keys-cited_keys)}; bibliography={len(bib_keys)}")

for figure in range(1,8):
    check(f"Figure {figure} cited and supplied", f"Figure {figure}." in MAIN and (BASE/f"figures/figure_{figure:02d}.png").exists(), "caption/reference and PNG present")
for table in range(1,10):
    check(f"Table {table} present", f"**Table {table}." in MAIN, "numbered main-text table")

anonymous_files = [BASE/"manuscript/main.md", BASE/"supplement/main.md"] + list((BASE/"supplement/evidence").glob("*"))
identity_terms = ["Muhammed Yusuf Batur","Mehmet Göktürk","myusuf.batur@","Rumeli University","Gebze Technical University"]
identity_hits=[]
for path in anonymous_files:
    if path.is_file():
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        for term in identity_terms:
            if term.lower() in text.lower(): identity_hits.append(f"{path.relative_to(BASE)}:{term}")
check("Anonymous text/evidence identity scan", not identity_hits, f"hits={identity_hits}")

for docx_rel in ["manuscript/anonymous_manuscript.docx","supplement/supplementary_material.docx"]:
    path=BASE/docx_rel
    hits=[]
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".xml"):
                text=archive.read(name).decode("utf-8",errors="ignore")
                hits.extend(term for term in identity_terms if term.lower() in text.lower())
    check(f"Anonymous DOCX identity scan: {docx_rel}", not hits, f"hits={sorted(set(hits))}")

forbidden_context = ["MDPI", "Round 2", "user-approved", "hostile review", "paid API"]
context_hits=[]
for rel in ["manuscript/main.md","supplement/main.md"]:
    text=(BASE/rel).read_text(encoding="utf-8-sig",errors="ignore")
    context_hits += [f"{rel}:{term}" for term in forbidden_context if term.lower() in text.lower()]
check("Internal-workflow language absent", not context_hits, f"hits={context_hits}")

deliverables = [
 "manuscript/final_submission.pdf","manuscript/anonymous_manuscript.docx","manuscript/main.tex","manuscript/references.bib",
 "supplement/Supplementary_Material.pdf","supplement/supplementary_material.docx",
 "declarations/TITLE_PAGE.docx","highlights/HIGHLIGHTS.docx","cover_letter/COVER_LETTER.pdf",
]
for rel in deliverables:
    path=BASE/rel
    check(f"Deliverable exists: {rel}", path.exists() and path.stat().st_size>0, f"{path.stat().st_size if path.exists() else 0} bytes")

checks.append({"check":"DOCX visual render","passed":False,"severity":"environmental","evidence":"Bundled renderer cannot run because LibreOffice soffice.exe is unavailable; equivalent PDFs were rendered and inspected."})
open_decisions = len(re.findall(r"awaiting|pending|unapproved|must be confirmed|confirmation required", "\n".join(p.read_text(encoding="utf-8-sig") for p in (BASE/"declarations").glob("*.md")), re.I))
checks.append({"check":"Author decisions complete","passed":open_decisions==0,"severity":"submission_blocker","evidence":f"{open_decisions} unresolved status markers across declaration drafts"})

manifest=[]
for rel in deliverables + ["supplement/main.md"] + [str(p.relative_to(BASE)).replace("\\","/") for p in sorted((BASE/"supplement/tables").glob("*.csv"))]:
    path=BASE/rel
    if path.exists(): manifest.append({"path":rel,"bytes":path.stat().st_size,"sha256":sha256(path.read_bytes()).hexdigest()})
(BASE/"PACKAGE_MANIFEST.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

required_failures=[c for c in checks if not c["passed"] and c["severity"]=="required"]
status={"scientific_and_format_qc":"PASS" if not required_failures else "FAIL", "submission_ready":False if any(not c["passed"] and c["severity"]=="submission_blocker" for c in checks) else not required_failures,
        "checks":checks,"manifest_entries":len(manifest)}
(BASE/"qc/FINAL_QC.json").write_text(json.dumps(status,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
lines=["# Final quality-control report","",f"Scientific and format QC: **{status['scientific_and_format_qc']}**",f"Submission ready: **{'YES' if status['submission_ready'] else 'NO — author decisions remain open'}**","","| Check | Result | Severity | Evidence |","| --- | --- | --- | --- |"]
for item in checks:
    evidence=item["evidence"].replace("|","\\|")
    lines.append(f"| {item['check']} | {'PASS' if item['passed'] else 'OPEN/FAIL'} | {item['severity']} | {evidence} |")
(BASE/"qc/FINAL_QC.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"scientific_and_format_qc":status["scientific_and_format_qc"],"submission_ready":status["submission_ready"],"required_failures":len(required_failures),"checks":len(checks),"manifest_entries":len(manifest)},indent=2))
raise SystemExit(1 if required_failures else 0)
