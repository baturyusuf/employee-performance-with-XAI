"""Remove only regenerated visual/compilation intermediates inside this package."""
from pathlib import Path
import shutil

base = Path(__file__).resolve().parents[1]
targets = [
    "qc/build", "qc/render_docs", "qc/render_main", "qc/render_supp", "qc/render_supp_v2", "qc/render_supp_v3",
    "qc/final_supp_page-03.png", "qc/latest_build.log",
    "manuscript/main.aux", "manuscript/main.log", "manuscript/main.pdf",
    "supplement/supplement.aux", "supplement/supplement.log", "supplement/supplement.pdf",
]
targets += [str(path.relative_to(base)) for path in (base / "qc").glob("*_contact_*.png")]
targets += [str(path.relative_to(base)) for path in (base / "qc").glob("suppcheck-*.png")]
targets += [str(path.relative_to(base)) for path in base.rglob("__pycache__")]
removed=[]
for relative in targets:
    target=(base/relative).resolve()
    target.relative_to(base.resolve())
    if target.is_dir(): shutil.rmtree(target)
    elif target.exists(): target.unlink()
    removed.append(relative)
print({"removed":len(removed),"base":str(base)})
