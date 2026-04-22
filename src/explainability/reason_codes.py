from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.config import SETTINGS


FEATURE_LABELS_TR = {
    "EmpEnvironmentSatisfaction": "çalışma ortamı memnuniyeti",
    "EmpLastSalaryHikePercent": "son maaş artış oranı",
    "YearsSinceLastPromotion": "son terfiden bu yana geçen süre",
    "EmpDepartment": "departman",
    "ExperienceYearsInCurrentRole": "mevcut rolde deneyim süresi",
    "EmpWorkLifeBalance": "iş-yaşam dengesi",
    "ExperienceYearsAtThisCompany": "şirkette toplam çalışma süresi",
    "EmpJobInvolvement": "işe katılım düzeyi",
    "YearsWithCurrManager": "mevcut yönetici ile çalışma süresi",
    "EmpJobRole": "iş rolü",
    "BusinessTravelFrequency": "iş seyahati sıklığı",
    "TotalWorkExperienceInYears": "toplam iş deneyimi",
    "DistanceFromHome": "eve uzaklık",
    "TrainingTimesLastYear": "geçen yıl alınan eğitim sayısı",
    "NumCompaniesWorked": "çalışılan şirket sayısı",
    "OverTime": "fazla mesai durumu",
    "EmpJobSatisfaction": "iş memnuniyeti",
    "EmpRelationshipSatisfaction": "ilişki memnuniyeti",
    "EmpEducationLevel": "eğitim seviyesi",
    "EducationBackground": "eğitim geçmişi",
    "Age": "yaş",
    "EmpHourlyRate": "saatlik ücret",
    "EmpJobLevel": "iş seviyesi",
    "Attrition": "işten ayrılma durumu",
    "MaritalStatus": "medeni durum",
    "Gender": "cinsiyet",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def get_latest_sample_dir(local_root: Path) -> Path:
    sample_dirs = [p for p in local_root.glob("sample_*") if p.is_dir()]
    if not sample_dirs:
        raise FileNotFoundError(
            f"No sample directories found under: {local_root}\n"
            f"Please run shap_local first."
        )
    sample_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sample_dirs[0]


def get_sample_dir(local_root: Path, sample_index: Optional[int]) -> Path:
    if sample_index is None:
        return get_latest_sample_dir(local_root)

    sample_dir = local_root / f"sample_{sample_index}"
    if not sample_dir.exists():
        raise FileNotFoundError(
            f"Sample directory not found: {sample_dir}\n"
            f"Run shap_local for this sample first."
        )
    return sample_dir


def load_local_explanation_files(
    sample_dir: Path,
    class_label: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Path, Path]:
    json_files = sorted(sample_dir.glob("local_explanation_class_*.json"))
    csv_files = sorted(sample_dir.glob("local_explanation_class_*.csv"))

    if not json_files or not csv_files:
        raise FileNotFoundError(
            f"Local explanation files not found in: {sample_dir}\n"
            f"Please run shap_local first."
        )

    if class_label is None:
        # pick the first available pair
        json_path = json_files[0]
        csv_path = csv_files[0]
    else:
        json_path = sample_dir / f"local_explanation_class_{class_label}.json"
        csv_path = sample_dir / f"local_explanation_class_{class_label}.csv"

        if not json_path.exists() or not csv_path.exists():
            raise FileNotFoundError(
                f"Expected local explanation files for class {class_label} not found in {sample_dir}"
            )

    local_df = pd.read_csv(csv_path)
    meta = load_json(json_path)
    return local_df, meta, csv_path, json_path


def to_readable_feature_name(feature: str) -> str:
    return FEATURE_LABELS_TR.get(feature, feature)


def describe_feature_value(feature: str, value: Any) -> str:
    if pd.isna(value):
        return "değer eksik"

    try:
        numeric_value = float(value)
        is_numeric = True
    except Exception:
        numeric_value = None
        is_numeric = False

    if feature == "EmpEnvironmentSatisfaction" and is_numeric:
        mapping = {
            1.0: "çok düşük",
            2.0: "düşük",
            3.0: "orta-iyi",
            4.0: "yüksek",
        }
        return f"seviye: {mapping.get(numeric_value, value)}"

    if feature == "EmpWorkLifeBalance" and is_numeric:
        mapping = {
            1.0: "çok zayıf",
            2.0: "zayıf",
            3.0: "dengeli",
            4.0: "çok iyi",
        }
        return f"seviye: {mapping.get(numeric_value, value)}"

    if feature == "EmpJobInvolvement" and is_numeric:
        mapping = {
            1.0: "çok düşük",
            2.0: "düşük",
            3.0: "iyi",
            4.0: "çok iyi",
        }
        return f"seviye: {mapping.get(numeric_value, value)}"

    if feature == "YearsSinceLastPromotion" and is_numeric:
        return f"{int(numeric_value)} yıl"

    if feature == "ExperienceYearsInCurrentRole" and is_numeric:
        return f"{int(numeric_value)} yıl"

    if feature == "ExperienceYearsAtThisCompany" and is_numeric:
        return f"{int(numeric_value)} yıl"

    if feature == "YearsWithCurrManager" and is_numeric:
        return f"{int(numeric_value)} yıl"

    if feature == "TotalWorkExperienceInYears" and is_numeric:
        return f"{int(numeric_value)} yıl"

    if feature == "EmpLastSalaryHikePercent" and is_numeric:
        return f"%{numeric_value:.0f}"

    if feature == "TrainingTimesLastYear" and is_numeric:
        return f"{int(numeric_value)} kez"

    return str(value)


def build_reason_sentence(feature: str, value: Any, shap_value: float) -> str:
    fname = to_readable_feature_name(feature)
    value_desc = describe_feature_value(feature, value)
    positive = shap_value >= 0

    # Feature-specific templates
    if feature == "EmpEnvironmentSatisfaction":
        return (
            f"{fname.capitalize()} ({value_desc}) performans tahminini yukarı çekmiştir."
            if positive else
            f"{fname.capitalize()} ({value_desc}) performans tahminini aşağı çekmiştir."
        )

    if feature == "EmpLastSalaryHikePercent":
        return (
            f"{fname.capitalize()} ({value_desc}) olumlu bir katkı sağlamıştır."
            if positive else
            f"{fname.capitalize()} ({value_desc}) tahmini zayıflatıcı yönde etki etmiştir."
        )

    if feature == "YearsSinceLastPromotion":
        return (
            f"{fname.capitalize()} ({value_desc}) bu sınıf lehine etki etmiştir."
            if positive else
            f"{fname.capitalize()} ({value_desc}) tahmini aşağı yönlü etkilemiştir."
        )

    if feature == "EmpDepartment":
        return (
            f"{fname.capitalize()} bilgisi ({value_desc}) modelde bu sınıf yönünde pozitif katkı üretmiştir."
            if positive else
            f"{fname.capitalize()} bilgisi ({value_desc}) modelde negatif katkı üretmiştir."
        )

    if feature == "EmpWorkLifeBalance":
        return (
            f"{fname.capitalize()} ({value_desc}) performans tahminini desteklemiştir."
            if positive else
            f"{fname.capitalize()} ({value_desc}) performans tahminini olumsuz etkilemiştir."
        )

    if feature == "EmpJobInvolvement":
        return (
            f"{fname.capitalize()} ({value_desc}) olumlu bir sinyal vermiştir."
            if positive else
            f"{fname.capitalize()} ({value_desc}) yetersiz bir sinyal vermiştir."
        )

    # Generic template
    return (
        f"{fname.capitalize()} ({value_desc}) model kararına pozitif katkı sağlamıştır."
        if positive else
        f"{fname.capitalize()} ({value_desc}) model kararına negatif katkı sağlamıştır."
    )


def build_reason_codes(
    local_df: pd.DataFrame,
    top_k_positive: int = 4,
    top_k_negative: int = 4,
    min_abs_shap: float = 0.005,
) -> Dict[str, List[Dict[str, Any]]]:
    filtered_df = local_df[local_df["abs_shap_value"] >= min_abs_shap].copy()

    positive_df = (
        filtered_df[filtered_df["shap_value"] > 0]
        .sort_values(by="abs_shap_value", ascending=False)
        .head(top_k_positive)
    )

    negative_df = (
        filtered_df[filtered_df["shap_value"] < 0]
        .sort_values(by="abs_shap_value", ascending=False)
        .head(top_k_negative)
    )

    positive_codes: List[Dict[str, Any]] = []
    negative_codes: List[Dict[str, Any]] = []

    for _, row in positive_df.iterrows():
        positive_codes.append({
            "feature": row["feature"],
            "feature_label_tr": to_readable_feature_name(row["feature"]),
            "feature_value": row["feature_value"],
            "feature_value_description": describe_feature_value(row["feature"], row["feature_value"]),
            "shap_value": float(row["shap_value"]),
            "sentence_tr": build_reason_sentence(
                feature=row["feature"],
                value=row["feature_value"],
                shap_value=float(row["shap_value"]),
            ),
        })

    for _, row in negative_df.iterrows():
        negative_codes.append({
            "feature": row["feature"],
            "feature_label_tr": to_readable_feature_name(row["feature"]),
            "feature_value": row["feature_value"],
            "feature_value_description": describe_feature_value(row["feature"], row["feature_value"]),
            "shap_value": float(row["shap_value"]),
            "sentence_tr": build_reason_sentence(
                feature=row["feature"],
                value=row["feature_value"],
                shap_value=float(row["shap_value"]),
            ),
        })

    return {
        "positive_reason_codes": positive_codes,
        "negative_reason_codes": negative_codes,
    }


def build_executive_summary(meta: Dict[str, Any], reason_codes: Dict[str, List[Dict[str, Any]]]) -> str:
    predicted_class = meta["predicted_class"]
    true_class = meta["true_class"]
    prob = meta["predicted_probability_for_predicted_class"]

    pos_lines = [f"- {item['sentence_tr']}" for item in reason_codes["positive_reason_codes"]]
    neg_lines = [f"- {item['sentence_tr']}" for item in reason_codes["negative_reason_codes"]]

    text = []
    text.append(f"Model bu çalışanı {predicted_class} performans sınıfında tahmin etti.")
    text.append(f"Gerçek sınıf: {true_class}.")
    text.append(f"Tahmin güveni: {prob:.4f}.")
    text.append("")
    text.append("Tahmini yukarı çeken başlıca nedenler:")
    text.extend(pos_lines if pos_lines else ["- Belirgin pozitif neden bulunamadı."])
    text.append("")
    text.append("Tahmini aşağı çeken başlıca nedenler:")
    text.extend(neg_lines if neg_lines else ["- Belirgin negatif neden bulunamadı."])

    return "\n".join(text)


def run_reason_code_generation(
    sample_index: Optional[int] = None,
    class_label: Optional[int] = None,
    top_k_positive: int = 4,
    top_k_negative: int = 4,
    min_abs_shap: float = 0.005,
) -> Dict[str, Any]:
    local_root = SETTINGS.reports_dir / "xai" / "local"
    sample_dir = get_sample_dir(local_root=local_root, sample_index=sample_index)

    local_df, meta, csv_path, json_path = load_local_explanation_files(
        sample_dir=sample_dir,
        class_label=class_label,
    )

    reason_codes = build_reason_codes(
        local_df=local_df,
        top_k_positive=top_k_positive,
        top_k_negative=top_k_negative,
        min_abs_shap=min_abs_shap,
    )

    summary_text = build_executive_summary(meta=meta, reason_codes=reason_codes)

    output = {
        "sample_index": meta["sample_index"],
        "true_class": meta["true_class"],
        "predicted_class": meta["predicted_class"],
        "explained_class": meta["explained_class"],
        "predicted_probability_for_predicted_class": meta["predicted_probability_for_predicted_class"],
        "source_files": {
            "local_explanation_csv": str(csv_path),
            "local_explanation_json": str(json_path),
        },
        "reason_codes": reason_codes,
        "summary_text_tr": summary_text,
    }

    explained_class = meta["explained_class"]

    save_json(
        output,
        sample_dir / f"reason_codes_class_{explained_class}.json",
    )
    save_text(
        summary_text,
        sample_dir / f"reason_codes_class_{explained_class}.txt",
    )

    print("\n=== REASON CODE GENERATION COMPLETE ===")
    print(f"Sample index: {meta['sample_index']}")
    print(f"Predicted class: {meta['predicted_class']}")
    print(f"Explained class: {meta['explained_class']}")
    print("\nSummary:\n")
    print(summary_text)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate human-readable reason codes from local SHAP output.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Sample index to process. If omitted, latest sample directory is used.",
    )
    parser.add_argument(
        "--class-label",
        type=int,
        default=None,
        help="Optional explained class label. If omitted, first available local explanation is used.",
    )
    parser.add_argument(
        "--top-k-positive",
        type=int,
        default=4,
        help="Number of top positive reason codes to keep.",
    )
    parser.add_argument(
        "--top-k-negative",
        type=int,
        default=4,
        help="Number of top negative reason codes to keep.",
    )
    parser.add_argument(
        "--min-abs-shap",
        type=float,
        default=0.005,
        help="Minimum absolute SHAP value to include in reason codes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_reason_code_generation(
        sample_index=args.sample_index,
        class_label=args.class_label,
        top_k_positive=args.top_k_positive,
        top_k_negative=args.top_k_negative,
        min_abs_shap=args.min_abs_shap,
    )