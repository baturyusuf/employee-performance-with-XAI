"""Record pinned runtime, available distribution notices, and safe source paths.

This prepares local metadata only. It does not create a release or software
licence and does not acquire or copy data, predictions, models, or Git history.
"""
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/submission_eswa/reproducibility'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, records):
    with (OUT / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    normalize = lambda value: re.sub(r'[-_.]+', '-', value).lower()
    installed = {normalize(dist.metadata['Name']): dist for dist in metadata.distributions() if dist.metadata['Name']}
    lock = dict(re.findall(r'^([\w.-]+)==([^\s;]+)', (ROOT / 'constraints/py314-lock.txt').read_text(), re.M))
    contract = json.loads((ROOT / 'configs/dependency_contract.yaml').read_text())['dependency_contract']
    core = set(map(normalize, contract['groups']['core']['direct_packages']))
    records = []
    for name, version in sorted(lock.items()):
        dist = installed.get(normalize(name))
        md = dist.metadata if dist else {}
        classifiers = md.get_all('Classifier', []) if dist else []
        license_fields = [md.get('License-Expression', ''), md.get('License', '')] + [c for c in classifiers if c.startswith('License ::')]
        notice_records = []
        for member in dist.files or [] if dist else []:
            if re.search(r'(^|/)(licenses?|copying|notice)([./_-]|$)', str(member), re.I):
                path = Path(dist.locate_file(member))
                if path.is_file():
                    notice_records.append({'distribution_relative_path': str(member).replace('\\', '/'), 'sha256': digest(path)})
        concise = md.get('License-Expression') or '; '.join(c.removeprefix('License :: ') for c in classifiers if c.startswith('License ::')) or (md.get('License') or '').split('\n')[0]
        records.append({
            'package': name, 'locked_version': version, 'core_direct': normalize(name) in core,
            'installed_version': dist.version if dist else '',
            'version_status': ('MATCH' if dist.version == version else 'MISMATCH') if dist else 'NOT_INSTALLED_OR_METADATA_ABSENT',
            'declared_licence_metadata': concise,
            'licence_metadata_sha256': hashlib.sha256('\n'.join(license_fields).encode()).hexdigest() if dist else '',
            'bundled_notice_files': json.dumps(notice_records, sort_keys=True),
            'review_scope': 'local installed metadata only; no legal compatibility conclusion',
        })
    write_csv('DEPENDENCY_LICENSE_INVENTORY.csv', records)
    acquisition = json.loads((ROOT / 'configs/data_acquisition.yaml').read_text())['data_acquisition']['physical_datasets']
    rights = {row['dataset_key']: row for row in []}
    datasets = []
    for name, spec in acquisition.items():
        path = ROOT / spec['local_path']
        datasets.append({'dataset': name, 'relative_local_path': spec['local_path'], 'expected_sha256': spec['expected_sha256'], 'observed_sha256': digest(path) if path.is_file() else '', 'local_hash_status': ('MATCH' if digest(path) == spec['expected_sha256'] else 'MISMATCH') if path.is_file() else 'NOT_PRESENT', 'raw_redistribution': 'EXCLUDED', 'automatic_download': spec['automatic_download_allowed']})
    write_csv('DATASET_HASHES.csv', datasets)
    allowed_roots = ['src', 'tools', 'tests', 'configs', 'constraints', 'manuscript/mdpi_information/assets', 'manuscript/mdpi_information/phase5b_figures', 'reports/research_log/major_revision_v3', 'reports/research_log/major_revision_round2', 'reports/research_log/finalization_v2']
    allowed_files = ['README.md', 'AGENTS.md', '.gitattributes', '.gitignore', 'environment.yml', 'requirements.txt', 'requirements-core.txt', 'requirements-dev.txt', 'requirements-legacy-optional.txt', 'requirements-supplementary.txt', 'data/README.md', 'data/external/hrdataset_v14/dataset_card.md', 'data/external/hrdataset_v14/schema_mapping.json', 'data/external/ibm_hr_analytics/dataset_card.md', 'data/external/ibm_hr_analytics/schema_mapping.json', 'data/external/employee_turnover/dataset_card.md', 'data/external/employee_turnover/schema_mapping.json']
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    chosen = [name for name in tracked if name and (name in allowed_files or any(name.startswith(root + '/') for root in allowed_roots))]
    records_source = []
    forbidden = re.compile(r'(^|/)(\.git|myenv|__pycache__)(/|$)|\.(joblib|pkl|pickle|parquet|xls|xlsx)$|(^|/)(oof_predictions|raw_oof_predictions|calibration_training_oof|fold_assignments|local_shap)\.csv$', re.I)
    for name in chosen:
        assert not forbidden.search(name), name
        assert not name.startswith(('data/raw/', 'data/interim/', 'data/processed/'))
        path = ROOT / name
        assert path.is_file(), name
        records_source.append({'path': name, 'size_bytes': path.stat().st_size, 'working_file_sha256': digest(path)})
    payload = {
        'status': 'LOCAL_SOURCE_MANIFEST_ONLY_NOT_A_RELEASE',
        'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        'scope': 'tracked scientific source/config/test/aggregate evidence paths at the recorded source commit; ESWA draft generated files excluded',
        'hash_scope': 'exact current filesystem bytes; checkout newline normalization may differ from Git blobs',
        'python': sys.version.split()[0], 'lock_sha256': digest(ROOT / 'constraints/py314-lock.txt'),
        'locked_packages': len(lock), 'core_direct_packages': len(core),
        'core_packages_without_installed_metadata': [r['package'] for r in records if r['core_direct'] and not r['installed_version']],
        'installed_version_mismatches': [r['package'] for r in records if r['version_status'] == 'MISMATCH'],
        'software_licence_selected': False, 'raw_data_included': False, 'employee_predictions_included': False, 'models_included': False, 'git_history_included': False,
        'archive_created': False, 'release_created': False, 'allowlist_roots': allowed_roots, 'allowlist_files': allowed_files,
        'file_count': len(records_source), 'total_bytes': sum(r['size_bytes'] for r in records_source), 'files': records_source,
    }
    (OUT / 'SOURCE_MANIFEST.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in payload.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
