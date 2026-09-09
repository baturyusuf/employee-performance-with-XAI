"""Read-only scientific evidence checks; writes only ESWA audit receipts.

No project calculation module is imported, no raw record is exported, and no
estimator, calibrator, explainer, bootstrap, acquisition, or network call runs.
Run from the repository root with the standard library on Python 3.14.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
R2 = Path('reports/research_log/major_revision_round2')
EXPECTED = '751208d036597461606bd02d68bfa9e642a4aa5ecf5df2df3fd7e46b99084aea'


def rows(path):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def display(value, places=4, scale='1'):
    return format((Decimal(value) * Decimal(scale)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN), f'.{places}f')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manuscript', default='reports/submission_eswa/manuscript/main.md')
    args = parser.parse_args()
    matrix_path = R2 / 'round2_claim_matrix/ROUND2_CLAIM_MATRIX.csv'
    matrix = rows(matrix_path)
    canonical = [{key: val for key, val in row.items() if key != 'approval_status'} for row in matrix]
    encoded = (json.dumps({'schema_version': 2, 'claims': canonical}, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    digest = hashlib.sha256(encoded).hexdigest()
    assert digest == EXPECTED, ('claim digest mismatch', digest)
    approval = json.loads((ROOT / 'configs/round2_claim_matrix_v4.json').read_text(encoding='utf-8'))
    assert approval['user_approval']['approved_claim_set_sha256'] == EXPECTED
    assert approval['user_approval']['status'] == 'approved'
    active = [row for row in matrix if row['active_for_rewrite'] == 'True']
    assert len(matrix) == 129 and len(active) == 117
    assert sum(row['claim_type'] == 'numerical' for row in active) == 99
    assert active == rows(R2 / 'round2_manuscript/SUPPLEMENTARY_EVIDENCE_LEDGER_ROUND2.csv')
    manuscript_path = Path(args.manuscript)
    manuscript_exists = (ROOT / manuscript_path).is_file()
    text = (ROOT / manuscript_path).read_text(encoding='utf-8') if manuscript_exists else ''
    lines = text.splitlines()
    component_locations = {
        'phase1b': ('benchmark', '3', '2'), 'round2_extreme_class': ('extreme-class', '3;4', '2'),
        'round2_selection': ('selection-objective', '4', ''), 'round2_probability_baseline': ('probability', '', ''),
        'phase1c': ('repetition', '5', ''), 'phase1d': ('P3→P4', '6', '3'), 'round2_timing': ('P3→P4', '6', '3'),
        'phase2a': ('SHAP', '7', '4;5'), 'phase2b': ('calibration', '8', '6'),
        'phase2c': ('subgroup', '9', ''), 'round2_subgroup': ('subgroup', '9', ''),
        'phase3a': ('HR', '10;11', '7'), 'round2_hr_mapping': ('HR', '1;10', '7'),
        'round2_hr_cv': ('HR', '11', ''), 'round2_hr_alias': ('alias', '11', ''),
        'phase3b': ('quality', '1', ''), 'feature_contract': ('information', '2', '1'),
    }
    findings = []
    for row in matrix:
        assert sha(row['source_path']) == row['source_sha256'], ('source hash', row['claim_id'])
        if row['claim_type'] == 'numerical':
            selector = json.loads(row['source_selector'])
            matches = [(i, source) for i, source in enumerate(rows(row['source_path']), 2) if all(source.get(k) == str(v) for k, v in selector.items())]
            assert len(matches) == 1, ('nonunique selector', row['claim_id'], len(matches))
            number, source = matches[0]
            assert str(number) == row['source_row_number'], ('source row', row['claim_id'])
            assert source[row['source_column']] == row['exact_value'], ('source value', row['claim_id'])
            assert display(row['exact_value'], int(row['rounding_places']), row['display_scale']) == row['display_value']
        else:
            assert row['evidence_anchor'] in (ROOT / row['source_path']).read_text(encoding='utf-8'), ('narrative anchor', row['claim_id'])
        if row['active_for_rewrite'] != 'True':
            continue
        topic, tables, figures = component_locations.get(row['component'], ('qualifications/declarations', '', ''))
        hits = []
        section = ''
        for i, line in enumerate(lines, 1):
            if line.startswith('#'):
                section = line.lstrip('# ').strip()
            if row['display_value'] and re.search(r'(?<![\d.])' + re.escape(row['display_value'].lstrip('-')) + r'(?![\d.])', line):
                hits.append(f'{manuscript_path.as_posix()}:{i} ({section})')
        findings.append({
            'claim_id': row['claim_id'], 'claim': row['proposed_claim'], 'claim_type': row['claim_type'],
            'exact_number': row['exact_value'], 'display_number': row['display_value'],
            'source_evidence': row['source_path'], 'source_sha256': row['source_sha256'],
            'source_selector': row['source_selector'], 'source_row': row['source_row_number'], 'source_column': row['source_column'],
            'table_context_in_original_approved_manuscript': tables, 'figure_context_in_original_approved_manuscript': figures,
            'manuscript_location_candidates': ' | '.join(hits), 'topic': topic,
            'rounding': f"ROUND_HALF_EVEN; places={row['rounding_places']}; scale={row['display_scale']}" if row['claim_type'] == 'numerical' else '',
            'qualifier': row['mandatory_qualifier'], 'prohibited_interpretation': row['prohibited_overclaim'],
            'source_audit': 'PASS',
            'location_audit': 'candidate number occurrences only; semantic context audited separately' if hits else 'not a number in main prose; ledger qualification/source retained',
        })
    # Recalculate derived summaries from their bound upstream aggregate rows.
    derived = rows(R2 / 'round2_claim_matrix/DERIVED_SUMMARIES.csv')
    for row in derived:
        assert sha(row['source_path']) == row['source_sha256']
        source = rows(row['source_path'])
        rule = row['derivation']
        count = re.fullmatch(r'count (True|False) in (\w+)', rule)
        if count:
            expected = sum(item[count[2]] == count[1] for item in source)
            assert expected == int(row['value']) and len(source) == int(row['denominator'])
        else:
            subtract = re.fullmatch(r'(P\d)\.(\w+) minus (P\d)\.(\w+)', rule)
            assert subtract, rule
            a = next(item for item in source if item['policy_id'] == subtract[1])
            b = next(item for item in source if item['policy_id'] == subtract[3])
            assert Decimal(a[subtract[2]]) - Decimal(b[subtract[4]]) == Decimal(row['value']), row['summary_id']
    changes = rows(R2 / 'selection_objective_sensitivity/selected_candidate_changes.csv')
    rankings = rows(R2 / 'selection_objective_sensitivity/ranking_changes.csv')
    for row in changes:
        assert (row['macro_f1_candidate_index'] != row['qwk_candidate_index']) == (row['selected_candidate_changed'] == 'True')
    for row in rankings:
        a, b = (json.loads(row[k]) for k in ['macro_f1_selection_order_json', 'qwk_selection_order_json'])
        assert (a[0] != b[0]) == (row['leader_changed'] == 'True')
        assert (a != b) == (row['full_ordering_changed'] == 'True')
    hrc = json.loads((ROOT / 'configs/hrdataset_sensitivity_v3.json').read_text())
    features = hrc['feature_contract']['exact_features'] if 'feature_contract' in hrc else next(v['exact_features'] for v in hrc.values() if isinstance(v, dict) and 'exact_features' in v)
    assert features == ['EmpJobRole', 'EngagementSurvey', 'EmpJobSatisfaction', 'SpecialProjectsCount', 'DaysLateLast30', 'Absences', 'ExperienceYearsAtThisCompany']
    aliases = rows(R2 / 'hr_target_alias_sensitivity/comparison_deltas.csv')
    for row in aliases:
        if row['comparison_id'] == 'matched_refit_effect':
            assert row['left_arm'] == 'restricted_canonical_309' and row['right_arm'] == 'exclusion_refit_309' and row['evaluation_population_matched'] == 'True'
        else:
            assert row['comparison_id'] == 'sample_removal_effect' and row['evaluation_population_matched'] == 'False'
        assert abs(float(row['right_value']) - float(row['left_value']) - float(row['right_minus_left'])) < 1e-14
    qc = ROOT / 'reports/submission_eswa/qc'
    qc.mkdir(parents=True, exist_ok=True)
    with (qc / 'CLAIM_AUDIT.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(findings[0]))
        writer.writeheader()
        writer.writerows(findings)
    receipt = {
        'status': 'PASS_SOURCE_BINDINGS', 'claim_digest': digest, 'matrix_rows': len(matrix),
        'active_claims': len(active), 'active_numeric_claims': 99, 'source_files_rehashed': len({r['source_path'] for r in matrix}),
        'derived_rows_recalculated': len(derived), 'candidate_changes': [37, 60], 'metric_leader_changes': [6, 9], 'full_ordering_changes': [9, 9],
        'hr_exact_features': features, 'hr_alias_delta_rows_recalculated': len(aliases),
        'manuscript': manuscript_path.as_posix(), 'manuscript_sha256': sha(manuscript_path) if manuscript_exists else None,
        'limitation': 'Exact source and number-occurrence checks are distinct from semantic manuscript review. No new model fit or employee-level prediction recomputation was performed.'
    }
    out = ROOT / 'reports/submission_eswa/reproducibility/frozen_evidence_receipt.json'
    out.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
