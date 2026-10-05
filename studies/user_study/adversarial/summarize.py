"""Verify the saved gates and freeze their candidate/compiler fingerprints."""
import hashlib
import json
from pathlib import Path

from ..tasks import TASKS
from ..validate import environment

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    current = environment()
    summaries, programs = [], []
    for task in TASKS:
        directory = ROOT / task.name
        inventory = json.loads((directory / 'inventory.json').read_text())
        results = {}
        for suffix in ('after', 'halo2'):
            path = ROOT / 'results' / f'{task.name}-{suffix}.json'
            report = json.loads(path.read_text())
            assert report['passed'] == report['total'] > 0, path
            for key in ('working_source_sha256', 'extension_sha256'):
                assert report['environment'][key] == current[key], (path, key)
            results[suffix] = report
        expected_programs = {Path(row['path']).name for row in inventory}
        assert {Path(row['path']).name for row in results['after']['rows']} == expected_programs
        assert len(results['after']['rows']) == len(inventory) * len(task.profiles)
        assert len(results['halo2']['rows']) == len(inventory)
        for item in inventory:
            path = directory / item['path']
            digest = sha(path)
            for report in results.values():
                rows = [r for r in report['rows'] if Path(r['path']).name == path.name]
                assert rows and all(r['source_sha256'] == digest for r in rows), path
                assert all(r['original_compiled'] and r['checked_compiled'] and r['negative_control'] for r in rows), path
            programs.append({'task': task.name, 'path': str(path.relative_to(ROOT)), 'sha256': digest})
        summaries.append({'task': task.name, 'programs': len(inventory),
            'profiles': results['after']['total'],
            'compiled_cases': sum(r['compiled_cases'] for r in results['after']['rows']),
            'negative_controls': sum(r['negative_control'] for r in results['after']['rows']),
            'verified_halo2_proofs': sum(r['compiled_cases'] for r in results['halo2']['rows'])})
    site = json.loads((ROOT / 'results/site-worker.json').read_text())
    assert site['passed'] == site['total'] == len(programs)
    expected = {(p['task'], Path(p['path']).name): p['sha256'] for p in programs}
    assert len(site['rows']) == len(expected)
    for row in site['rows']:
        assert row['source_sha256'] == expected[(row['task'], Path(row['path']).name)]
        assert row['result']['status'] == 'passed'
    for key in ('working_source_sha256', 'extension_sha256'):
        assert site['environment'][key] == current[key]
    for label in ('curated', 'pilot'):
        report = json.loads((ROOT / f'results/{label}-regression.json').read_text())
        assert report['passed'] == report['total'] > 0
        for key in ('working_source_sha256', 'extension_sha256'):
            assert report['environment'][key] == current[key]
    files = [ROOT / name for name in ('check.py','site_check.py','summarize.py')]
    files += sorted(ROOT.glob('*/inventory.json'))
    files += sorted(ROOT.glob('*/REPORT.md'))
    files += sorted(ROOT.glob('*/baseline*.json'))
    files += [ROOT / 'README.md']
    files += sorted((ROOT / 'results').glob('*'))
    manifest = {'environment': current,
        'review_model': 'gpt-6-astra', 'reviewer_assignment': 'one reviewer per task',
        'tasks': summaries, 'programs': programs,
        'site_programs': site['passed'],
        'site_cases': sum(len(r['result']['cases']) for r in site['rows']),
        'artifact_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files if p.is_file()}}
    (ROOT / 'final-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'programs': len(programs),
        'profiles': sum(r['profiles'] for r in summaries),
        'compiled_cases': sum(r['compiled_cases'] for r in summaries),
        'negative_controls': sum(r['negative_controls'] for r in summaries),
        'halo2_proofs': sum(r['verified_halo2_proofs'] for r in summaries),
        'site_cases': manifest['site_cases']}, indent=2))


if __name__ == '__main__':
    main()
