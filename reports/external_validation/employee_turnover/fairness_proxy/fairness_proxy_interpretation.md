# External Fairness and Proxy Audit: Human Resources Analytics / Employee Turnover

Minimum support threshold: 30
Audit attributes: EmpDepartment, SalaryBand, AverageMonthlyHours, LastEvaluation, PromotionLast5Years

These subgroup metrics are diagnostic model-governance evidence. They do not prove fairness, discrimination, or absence of discrimination.

## Largest Support-Filtered Gaps

- policy=with_last_evaluation, attribute=AverageMonthlyHours, metric=false_positive_rate, class=0.0: gap=1.0000
- policy=with_last_evaluation, attribute=AverageMonthlyHours, metric=precision, class=1.0: gap=1.0000
- policy=with_last_evaluation, attribute=AverageMonthlyHours, metric=true_positive_rate, class=1.0: gap=1.0000
- policy=with_last_evaluation, attribute=LastEvaluation, metric=false_positive_rate, class=0.0: gap=1.0000
- policy=with_last_evaluation, attribute=LastEvaluation, metric=precision, class=1.0: gap=1.0000
- policy=with_last_evaluation, attribute=LastEvaluation, metric=true_positive_rate, class=1.0: gap=1.0000
- policy=without_last_evaluation, attribute=AverageMonthlyHours, metric=true_positive_rate, class=1.0: gap=1.0000
- policy=without_last_evaluation, attribute=AverageMonthlyHours, metric=false_positive_rate, class=0.0: gap=1.0000
- policy=without_last_evaluation, attribute=LastEvaluation, metric=false_positive_rate, class=0.0: gap=1.0000
- policy=without_last_evaluation, attribute=AverageMonthlyHours, metric=precision, class=1.0: gap=1.0000
- policy=without_last_evaluation, attribute=LastEvaluation, metric=true_positive_rate, class=1.0: gap=1.0000
- policy=without_last_evaluation, attribute=LastEvaluation, metric=precision, class=1.0: gap=1.0000

## Small-Group Warnings

- AverageMonthlyHours=280: n=29, n_samples < 30
- AverageMonthlyHours=113: n=29, n_samples < 30
- AverageMonthlyHours=104: n=28, n_samples < 30
- AverageMonthlyHours=111: n=26, n_samples < 30
- AverageMonthlyHours=126: n=25, n_samples < 30
- AverageMonthlyHours=283: n=25, n_samples < 30
- AverageMonthlyHours=284: n=24, n_samples < 30
- AverageMonthlyHours=301: n=24, n_samples < 30
- AverageMonthlyHours=121: n=24, n_samples < 30
- AverageMonthlyHours=98: n=23, n_samples < 30
- AverageMonthlyHours=277: n=21, n_samples < 30
- AverageMonthlyHours=296: n=21, n_samples < 30
- AverageMonthlyHours=308: n=20, n_samples < 30
- AverageMonthlyHours=123: n=20, n_samples < 30
- AverageMonthlyHours=289: n=19, n_samples < 30
- AverageMonthlyHours=106: n=19, n_samples < 30
- AverageMonthlyHours=100: n=19, n_samples < 30
- AverageMonthlyHours=125: n=19, n_samples < 30
- AverageMonthlyHours=305: n=18, n_samples < 30
- AverageMonthlyHours=306: n=18, n_samples < 30
- AverageMonthlyHours=310: n=18, n_samples < 30
- AverageMonthlyHours=108: n=18, n_samples < 30
- AverageMonthlyHours=109: n=18, n_samples < 30
- AverageMonthlyHours=117: n=18, n_samples < 30
- AverageMonthlyHours=304: n=17, n_samples < 30
- AverageMonthlyHours=291: n=17, n_samples < 30
- AverageMonthlyHours=102: n=17, n_samples < 30
- AverageMonthlyHours=105: n=17, n_samples < 30
- AverageMonthlyHours=103: n=17, n_samples < 30
- AverageMonthlyHours=294: n=16, n_samples < 30
- AverageMonthlyHours=309: n=16, n_samples < 30
- AverageMonthlyHours=101: n=16, n_samples < 30
- AverageMonthlyHours=292: n=15, n_samples < 30
- AverageMonthlyHours=290: n=15, n_samples < 30
- AverageMonthlyHours=114: n=15, n_samples < 30
- AverageMonthlyHours=307: n=14, n_samples < 30
- AverageMonthlyHours=115: n=14, n_samples < 30
- AverageMonthlyHours=97: n=14, n_samples < 30
- AverageMonthlyHours=298: n=13, n_samples < 30
- AverageMonthlyHours=293: n=13, n_samples < 30
- AverageMonthlyHours=124: n=13, n_samples < 30
- AverageMonthlyHours=295: n=12, n_samples < 30
- AverageMonthlyHours=110: n=12, n_samples < 30
- AverageMonthlyHours=118: n=12, n_samples < 30
- AverageMonthlyHours=300: n=11, n_samples < 30
- AverageMonthlyHours=122: n=11, n_samples < 30
- AverageMonthlyHours=99: n=11, n_samples < 30
- AverageMonthlyHours=119: n=10, n_samples < 30
- AverageMonthlyHours=107: n=10, n_samples < 30
- AverageMonthlyHours=116: n=10, n_samples < 30
- AverageMonthlyHours=120: n=10, n_samples < 30
- AverageMonthlyHours=112: n=10, n_samples < 30
- AverageMonthlyHours=302: n=8, n_samples < 30
- AverageMonthlyHours=297: n=7, n_samples < 30
- AverageMonthlyHours=288: n=6, n_samples < 30
- AverageMonthlyHours=299: n=6, n_samples < 30
- AverageMonthlyHours=303: n=6, n_samples < 30
- AverageMonthlyHours=96: n=6, n_samples < 30
- LastEvaluation=0.36: n=22, n_samples < 30

## Required Claim Limits

- Removing direct group variables is not evidence that the model is fair.
- Department, role, salary, tenure, and evaluation-history features can act as organisational proxies.
- Any subgroup result with low support must be treated as unstable.
