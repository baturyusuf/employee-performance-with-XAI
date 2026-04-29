
# Q1 Upgrade Scripts

Bu paket mevcut projeye Q1 düzeyinde deneysel derinlik kazandırmak için hazırlanmıştır.

## Dosyalar

- `src/experiments/q1_experiments.py`
  - çoklu model benchmark
  - stratified k-fold cross-validation
  - imbalance benchmark
  - SHAP-top-k ve feature-group ablation
  - bootstrap CI + Wilcoxon/Holm istatistiksel test
  - computational complexity
  - formal fairness metrics
  - counterfactual evaluation

- `src/explainability/shap_extended.py`
  - beeswarm-like SHAP summary
  - waterfall-like local explanation
  - dependence plots
  - representative case tables

## Opsiyonel paketler

Bazı modeller/deneyler için:

```bash
pip install scipy imbalanced-learn xgboost lightgbm interpret
```

Bu paketler yoksa bazı modeller atlanır veya ilgili deney sınırlı çalışır.

## Önerilen çalıştırma sırası

Önce ana model ve SHAP:

```bash
python -m src.models.train_catboost --drop-sensitive
python -m src.explainability.shap_global
```

Sonra Q1 deneyleri:

```bash
python -m src.experiments.q1_experiments --task cv --drop-sensitive
python -m src.experiments.q1_experiments --task imbalance --drop-sensitive
python -m src.experiments.q1_experiments --task ablation --drop-sensitive
python -m src.experiments.q1_experiments --task stats --metric macro_f1
python -m src.experiments.q1_experiments --task stats --metric quadratic_weighted_kappa
python -m src.experiments.q1_experiments --task complexity
python -m src.experiments.q1_experiments --task fairness
python -m src.experiments.q1_experiments --task cf --max-samples 100
python -m src.explainability.shap_extended
```

Tek seferde çalıştırmak istersen:

```bash
python -m src.experiments.q1_experiments --task all --drop-sensitive
python -m src.explainability.shap_extended
```

## Çıktı klasörleri

- `reports/q1_experiments/cv_benchmark/`
- `reports/q1_experiments/imbalance/`
- `reports/q1_experiments/ablation/`
- `reports/q1_experiments/statistical_tests/`
- `reports/q1_experiments/complexity/`
- `reports/q1_experiments/counterfactual_evaluation/`
- `reports/fairness/formal/`
- `reports/xai/extended/`
