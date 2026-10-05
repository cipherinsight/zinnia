"""Run the frozen Astra candidates through the actual isolated study worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import tempfile

from ..site.content import cases
from ..site.execution import Runner
from ..validate import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task')
    args = parser.parse_args()
    base = Path(__file__).parent
    jobs = []
    for inventory in sorted(base.glob('*/inventory.json')):
        task = inventory.parent.name
        if args.task and task != args.task:
            continue
        for item in json.loads(inventory.read_text()):
            jobs.append((task, inventory.parent / item['path']))
    before = environment()
    rows = []
    with tempfile.TemporaryDirectory(prefix='astra-study-worker-') as directory:
        runner = Runner(directory)
        def run(job):
            task, path = job
            source = path.read_text()
            result = runner.execute('zinnia', task, source, cases(task) + cases(task, hidden=True))
            return {'task': task, 'path': str(path),
                    'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'result': result}
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run, job) for job in jobs]
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    if row['result']['status'] != 'passed':
                        print(row['path'], row['result']['status'], row['result'].get('message'), flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps({'environment': before, 'rows': rows,
                        'total': len(jobs), 'passed': sum(r['result']['status']=='passed' for r in rows)}, indent=2)+'\n')
        finally:
            runner.pool.shutdown()
    passed = sum(row['result']['status'] == 'passed' for row in rows)
    print(f'{passed}/{len(jobs)} study-worker submissions passed', flush=True)
    raise SystemExit(passed != len(jobs))


if __name__ == '__main__':
    main()
