"""Build anonymous ESWA review artifacts from the presentation-only Markdown."""
from pathlib import Path
import re, subprocess, shutil, sys, json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import black
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "reports/submission_eswa"
MAN = BASE / "manuscript"
SUP = BASE / "supplement"
TMP = BASE / "qc/build"
TMP.mkdir(parents=True, exist_ok=True)
TOOLS = Path.home() / "AppData/Local/Temp/codex-eswa-typesetting"
PANDOC = TOOLS / "pandoc-3.11/pandoc.exe"
TECTONIC = TOOLS / "tectonic.exe"
TITLE = "Auditing Intelligent Employee-Performance Prediction Systems Beyond Accuracy: A Traceable XAI Protocol"

def run(args, cwd=None):
    print("RUN", " ".join(map(str,args)))
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)

def cleaned(path, title):
    content = path.read_text(encoding="utf-8-sig")
    content = re.sub(r"^# .+?\n+", "", content, count=1)
    content = re.sub(r"\n## References\s*$", "\n", content)
    content = content.replace("(../figures/", "(figures/")
    out = TMP / (path.parent.name + "_render.md")
    yaml = f'---\ntitle: "{title}"\nauthor: []\ndate: ""\n---\n\n'
    out.write_text(yaml + content, encoding="utf-8")
    return out

main_render = cleaned(MAN / "main.md", TITLE)
sup_render = cleaned(SUP / "main.md", "Supplementary material")

common = ["--standalone", "--citeproc", "--bibliography", str(MAN/"references.bib"), "--csl", str(MAN/"apa.csl"), "--resource-path", str(BASE)]
run([PANDOC, main_render, *common, "--reference-location=block", "-o", MAN/"anonymous_manuscript.docx"])
run([PANDOC, sup_render, "--standalone", "--resource-path", str(BASE), "-o", SUP/"supplementary_material.docx"])
run([PANDOC, BASE/"declarations/TITLE_PAGE.md", "--standalone", "-o", BASE/"declarations/TITLE_PAGE.docx"])
run([PANDOC, BASE/"highlights/HIGHLIGHTS.md", "--standalone", "-o", BASE/"highlights/HIGHLIGHTS.docx"])

run([PANDOC, main_render, *common, "--reference-location=block", "-t", "latex", "-V", "documentclass=elsarticle", "-V", "classoption=review", "-V", "papersize=a4", "-o", MAN/"main.tex"])
run([PANDOC, sup_render, "--standalone", "-t", "latex", "-V", "documentclass=elsarticle", "-V", "classoption=review", "-V", "papersize=a4", "-o", SUP/"supplement.tex"])

main_tex = (MAN/"main.tex").read_text(encoding="utf-8")
main_tex = main_tex.replace("{figures/", "{../figures/")
main_tex = main_tex.replace("\\begin{document}", "\\begin{document}\n\\emergencystretch=3em")
main_tex = main_tex.replace("\\begin{longtable}", "\\begingroup\\scriptsize\n\\begin{longtable}")
main_tex = main_tex.replace("\\end{longtable}", "\\end{longtable}\n\\endgroup")
(MAN/"main.tex").write_text(main_tex, encoding="utf-8")

supp_tex = (SUP/"supplement.tex").read_text(encoding="utf-8")
supp_tex = supp_tex.replace("\\begin{document}", "\\begin{document}\n\\emergencystretch=4em\n\\setlength{\\tabcolsep}{2.5pt}")
supp_tex = supp_tex.replace("\\begin{longtable}", "\\begingroup\\scriptsize\n\\begin{longtable}")
supp_tex = supp_tex.replace("\\end{longtable}", "\\end{longtable}\n\\endgroup")
(SUP/"supplement.tex").write_text(supp_tex, encoding="utf-8")

shutil.copyfile(MAN/"elsarticle.cls", SUP/"elsarticle.cls")
run([TECTONIC, "--keep-logs", "--keep-intermediates", "--outdir", str(MAN), "main.tex"], cwd=MAN)
run([TECTONIC, "--keep-logs", "--keep-intermediates", "--outdir", str(SUP), "supplement.tex"], cwd=SUP)
if (MAN/"main.pdf").exists(): shutil.copyfile(MAN/"main.pdf", MAN/"final_submission.pdf")
if (SUP/"supplement.pdf").exists(): shutil.copyfile(SUP/"supplement.pdf", SUP/"Supplementary_Material.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="LetterBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=10, textColor=black, alignment=TA_LEFT))
styles["Title"].fontName="Helvetica-Bold"; styles["Title"].fontSize=17; styles["Title"].textColor=black
story=[Paragraph("Cover letter draft",styles["Title"]),Spacer(1,16)]
raw=(BASE/"cover_letter/COVER_LETTER.md").read_text(encoding="utf-8-sig")
for block in re.split(r"\n\s*\n",raw):
    block=block.strip()
    if not block or block.startswith("# "): continue
    block=block.replace("*Expert Systems with Applications*","<i>Expert Systems with Applications</i>")
    story.append(Paragraph(block,styles["LetterBody"]))
doc=SimpleDocTemplate(str(BASE/"cover_letter/COVER_LETTER.pdf"),pagesize=A4,rightMargin=55,leftMargin=55,topMargin=55,bottomMargin=55,title="Cover letter draft",author="")
doc.build(story)

print(json.dumps({"docx":4,"pdf":3,"tex":2},indent=2))
