"""Provisional circuit-size comparisons; includes uncounted and failed points."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


def summarize(audit: dict, counts: list[Path], zinnia_files: list[Path]) -> dict:
    # Caller orders corrected/recomputed Zinnia files last. Merge per backend
    # because heavy files can contain only Halo2 while Noir was counted earlier.
    zinnia = {}
    for path in zinnia_files:
        for group, entries in json.loads(path.read_text()).items():
            if group == 'config' or not isinstance(entries, dict):
                continue
            for case, targets in entries.items():
                if isinstance(targets, dict):
                    zinnia.setdefault(case, {}).update(targets)
    records = {}
    for path in counts:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            records[row['case'], row['system']] = row
    points = []
    for source in audit['points']:
        row = {k: source[k] for k in ['case', 'cap', 'system']}
        row['source_status'] = source['status']
        if source['status'] != 'rewritten':
            row['status'] = source['status']
        else:
            record = records.get((source['case'], source['system']), {})
            row['status'] = record.get('status', 'not_run')
            if record and record.get('output_sha256') != source['output_sha256']:
                row['status'] = 'stale_source'
            if row['status'] == 'ok':
                result = record['result']
                target = 'halo2' if source['system'] == 'halo2_hw' else 'noir'
                size = 'cells' if target == 'halo2' else 'gates'
                row['handwritten'] = (result['cells']['advice'] + result['cells']['lookup']
                                      if target == 'halo2' else result['gates'])
                compiler = zinnia.get(source['case'], {}).get(target, {})
                row['zinnia'] = compiler.get(size)
                if isinstance(row['zinnia'], (int, float)) and row['handwritten'] > 0:
                    row['ratio'] = row['zinnia'] / row['handwritten']
                else:
                    row['comparison_status'] = 'zinnia_count_missing'
        points.append(row)
    groups = []
    for cap in sorted({p['cap'] for p in points}):
        for system in ['halo2_hw', 'noir_hw']:
            selected = [p for p in points if p['cap'] == cap and p['system'] == system]
            ratios = [p['ratio'] for p in selected if p.get('ratio', 0) > 0]
            groups.append({'cap': cap, 'system': system,
                           'statuses': dict(Counter(p['status'] for p in selected)),
                           'comparable': len(ratios),
                           'geomean_zinnia_over_handwritten': math.exp(sum(map(math.log, ratios))/len(ratios)) if ratios else None,
                           'zinnia_smaller': sum(r < 1 for r in ratios),
                           'equal': sum(r == 1 for r in ratios),
                           'zinnia_larger': sum(r > 1 for r in ratios)})
    return {'note': 'Provisional local circuit counts, not timing. Source audit does not validate scaled outputs; '
                    'only CAP32 atax has A/B scaled output smoke validation in this run. '
                    'Existing Q32 errors and goodExpoMeasure numeric-domain limitations still apply.',
            'zinnia_inputs_in_override_order': [str(p) for p in zinnia_files],
            'groups': groups, 'points': points}


def main(args) -> int:
    report = summarize(json.loads(args.audit.read_text()), args.counts, args.zinnia)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    lines = ['# Scaled handwritten circuit counts', '', report['note'], '',
             '| CAP | Target | Counted | Compared | Zinnia / handwritten | Status counts |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for g in report['groups']:
        ratio = g['geomean_zinnia_over_handwritten']
        lines.append(f"| {g['cap']} | {g['system']} | {g['statuses'].get('ok', 0)} | {g['comparable']} | "
                     f"{format(ratio, '.4f') if ratio else '—'} | {g['statuses']} |")
    args.out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    return 0
