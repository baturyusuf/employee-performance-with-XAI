# Calibration Interpretation for Final Candidates

Protocol: outer 10-fold CV. Inside each training fold, an inner train/calibration split is used. Sigmoid and isotonic calibrators are fitted only on the inner calibration split, then evaluated on the outer test fold.

## Summary by Candidate

### `no_salary_hike_no_attrition`
- Best ECE: `sigmoid` (0.0599).
- Best log loss: `sigmoid` (0.4497).
- Best Brier: `sigmoid` (0.2561).
- Raw nested macro-F1/QWK: 0.6186 / 0.6514.

### `no_salary_hike_no_attrition_no_department`
- Best ECE: `isotonic` (0.0545).
- Best log loss: `sigmoid` (0.4551).
- Best Brier: `sigmoid` (0.2608).
- Raw nested macro-F1/QWK: 0.6137 / 0.6459.

### `no_salary_hike_no_attrition_no_department_no_job_role`
- Best ECE: `sigmoid` (0.0649).
- Best log loss: `sigmoid` (0.5119).
- Best Brier: `sigmoid` (0.3039).
- Raw nested macro-F1/QWK: 0.5589 / 0.5438.

## Required Answers

### Which candidate has the best probability quality?
The dashboard combines candidate-level metrics. For the primary candidate, the best average rank across log loss, Brier, and ECE is `sigmoid`.

### Does calibration improve probability quality?
Calibration is mixed unless it improves log loss, Brier, and ECE jointly. Do not select a method by ECE alone.

### Does calibration introduce overfitting or instability risk?
Yes, especially isotonic calibration because class 4 is small. Sigmoid is less flexible and usually safer with limited calibration data.

### Should the final model output calibrated probabilities, raw probabilities, or probability bands with warnings?
Use probability bands with calibration warnings. If `sigmoid` is used in tables, disclose the nested calibration protocol. Its macro-F1 difference versus raw in this protocol is 0.0190.
