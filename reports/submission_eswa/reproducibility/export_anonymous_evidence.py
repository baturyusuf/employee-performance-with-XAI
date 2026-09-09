"""Copy aggregate claim sources into an anonymous, numerically exact supplement."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/submission_eswa'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    claims = [row for row in read(ROOT / 'reports/research_log/major_revision_round2/SUPPLEMENTARY_EVIDENCE_LEDGER_ROUND2.csv') if row['claim_type'] == 'numerical']
    assert len(claims) == 99
    out = BASE / 'supplement/evidence'
    out.mkdir(parents=True, exist_ok=True)
    crosswalk, public, mapping = [], [], {}
    remove = {
        'DERIVED_SUMMARIES.csv': {'source_path', 'source_sha256'},
        'department_reconstructability_metrics.csv': {'run_id', 'config_hash', 'scientific_input_hash', 'fold_contract_hash', 'xgboost_model_set_sha256', 'dataset_sha256'},
    }
    forbidden_fields = {'sample_id', 'employee_id', 'EmpID', 'EmpNumber', 'Employee_Name', 'y_true', 'y_pred', 'outer_fold', 'inner_fold', 'prob_2', 'prob_3', 'prob_4', 'shap_value'}
    for number, name in enumerate(sorted({r['source_path'] for r in claims}), 1):
        source = ROOT / name
        original = read(source)
        assert original and not (set(original[0]) & forbidden_fields), name
        removed = remove.get(source.name, set())
        columns = [key for key in original[0] if key not in removed]
        filename = f'source_{number:02d}.csv'
        target = out / filename
        if removed:
            write(target, columns, [{key: row[key] for key in columns} for row in original])
        else:
            shutil.copyfile(source, target)
        assert all({key: old[key] for key in columns} == new for old, new in zip(original, read(target)))
        assert len(original) == len(read(target))
        content = target.read_text(encoding='utf-8-sig')
        assert not re.search(r'Yusuf|C:\\Users|github\.com/|AUTHOR|Codex|ChatGPT|reports/research_log', content, re.I), filename
        record = {'source_file': filename, 'source_sha256': sha(target), 'row_count': len(original)}
        public.append(record)
        mapping[name] = record
        crosswalk.append({**record, 'original_source_path': name, 'original_sha256': sha(source), 'removed_provenance_columns': ' | '.join(sorted(removed)), 'scientific_value_changes': 0, 'employee_level_rows_included': 0})
    ledger = []
    for claim in claims:
        item = mapping[claim['source_path']]
        source = read(out / item['source_file'])
        selector = json.loads(claim['source_selector'])
        matches = [row for row in source if all(row.get(k) == v for k, v in selector.items())]
        assert len(matches) == 1 and matches[0][claim['source_column']] == claim['exact_value'], claim['claim_id']
        ledger.append({
            'claim_id': claim['claim_id'], 'claim_text': claim['proposed_claim'], 'exact_value': claim['exact_value'], 'display_value': claim['display_value'],
            'source_file': item['source_file'], 'source_sha256': item['source_sha256'], 'source_selector': claim['source_selector'], 'source_row_number': claim['source_row_number'], 'source_column': claim['source_column'],
            'rounding': 'ROUND_HALF_EVEN', 'rounding_places': claim['rounding_places'], 'display_scale': claim['display_scale'],
            'qualifier': claim['mandatory_qualifier'], 'prohibited_interpretation': claim['prohibited_overclaim'],
        })
    write(out / 'NUMERICAL_EVIDENCE_LEDGER.csv', list(ledger[0]), ledger)
    write(out / 'SOURCE_INDEX.csv', list(public[0]), public)
    write(BASE / 'qc/ANONYMOUS_EVIDENCE_CROSSWALK.csv', list(crosswalk[0]), crosswalk)
    (out / 'README.md').write_text('''# Numerical evidence supplement

The numerical ledger contains 99 source-linked numerical claims. Each claim identifies an aggregate source file, SHA-256, an exact row selector and stored value, deterministic rounding, the necessary qualification, and the interpretation that the evidence does not support.

The 20 numbered source files contain aggregate scientific evidence only. All scientific rows and values are preserved. Internal project paths and generation-identifying provenance columns were removed from two source copies; the displayed source hashes identify these anonymous copies. The remaining 18 files are byte-identical source copies. Original-source identity is retained in the private editorial crosswalk outside this supplement.

These files contain no employee records, fold memberships, individual predictions, fitted models, author names, approval correspondence, or development logs. Class counts, subgroup names and support, aggregate metrics, descriptive repetition summaries, and uncertainty qualifications remain visible.

Numerical values are conditional on their stated dataset, model, selection rule, information policy, class, and validation design. Several claims occur only in the ledger or broader aggregate context; inclusion does not imply that every number is a headline result. The label-only baselines' historical probability cells remain in an aggregate source as preserved evidence, but are omitted from the manuscript's probabilistic comparison; the empirical-prior predictor supplies that reference.
''', encoding='utf-8')
    print(json.dumps({'numerical_claims': len(ledger), 'source_files': len(public), 'byte_copies': 18, 'provenance_column_only_views': 2, 'scientific_value_changes': 0, 'source_hashes_passed': True, 'unique_value_bindings_passed': True}))


if __name__ == '__main__':
    main()
