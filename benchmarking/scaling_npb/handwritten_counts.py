"""Serial, resumable circuit counts for source-audited scaled baselines.

Halo2 uses stats, Noir uses setup (compilation, verification key and gate count).
Neither runs proving or a timing benchmark. A memory-limiting runner is required.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(command: list[str], cwd: Path, env: dict, timeout: float, log: Path) -> dict:
    started = time.time()
    with log.open('w') as stream:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return {'status': 'timeout', 'reason': f'exceeded {timeout} seconds'}
    result = {'status': 'ok' if code == 0 else ('timeout' if code == 124 else 'failed'), 'returncode': code}
    if code != 0 and '--out' in command:
        # systemd-run may itself receive SIGTERM when the scope OOM-kills its
        # workload. Establish OOM from the matching scope journal, not a signal.
        output = command[command.index('--out') + 1]
        try:
            journal = subprocess.run(['journalctl', '--user', '--since', f'@{started:.0f}',
                                      '-o', 'json', '--no-pager'], capture_output=True, text=True, timeout=10)
            entries = [json.loads(line) for line in journal.stdout.splitlines()]
            units = {e.get('USER_UNIT') for e in entries
                     if f'--out {output}' in e.get('MESSAGE', '') and e.get('USER_UNIT')}
            evidence = [e for e in entries if e.get('USER_UNIT') in units]
            if any("result 'oom-kill'" in e.get('MESSAGE', '') for e in evidence):
                result.update(status='out_of_memory', reason='memory-limited scope reports oom-kill',
                              scope=sorted(units))
                log.with_suffix('.resource.json').write_text(json.dumps(evidence, indent=1) + '\n')
        except (OSError, ValueError, subprocess.SubprocessError):
            pass  # unavailable journal does not justify guessing an OOM
    if code < 0 and result['status'] == 'failed':
        result.update(status='signal_terminated', reason=f'terminated by signal {-code}; cause unconfirmed')
    return result


def main(args) -> int:
    audit = json.loads(args.audit.read_text())
    root = Path(audit['baseline_root'])
    baselines = root / 'benchmarking/baselines'
    points = [p for p in audit['points'] if p['cap'] == args.cap and p['status'] == 'rewritten'
              and (not args.cases or p['case'] in args.cases.split(','))]
    if not points:
        raise ValueError('no rewritten points selected')
    if any(p['system'] == 'halo2_hw' for p in points) and not args.halo2_bin:
        raise ValueError('--halo2-bin is required for Halo2 counts (build for this cap first)')
    compiled = {}
    if any(p['system'] == 'halo2_hw' for p in points):
        listed = subprocess.run([str(args.halo2_bin.resolve()), 'list', '--sources', 'true'],
                                capture_output=True, text=True, check=True, timeout=30)
        compiled = json.loads(listed.stdout).get('sources', {})
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            row = json.loads(line)
            old[(row['case'], row['system'])] = row
    env = dict(os.environ, MEMRUN_MAX=args.memory, RAYON_NUM_THREADS='2', OMP_NUM_THREADS='2',
               PYTHONPATH=str(baselines))
    # Invalidate successful cached counts when a gadget/driver or tool changes.
    tools = list((baselines / 'noir_hw').rglob('*.nr')) + list((baselines / 'hw_check').glob('*.py'))
    for tool in [env.get('NARGO', str(Path.home() / '.nargo/bin/nargo')),
                 env.get('BB', str(Path.home() / '.bb/bb'))]:
        if Path(tool).is_file():
            tools.append(Path(tool))
    noir_digest = hashlib.sha256(''.join(digest(p) for p in sorted(tools)).encode()).hexdigest()
    failures = 0
    for point in points:
        key = point['case'], point['system']
        path = Path(point['output'])
        row = {k: point[k] for k in ['case', 'cap', 'system', 'output_sha256']}
        if not path.exists() or digest(path) != point['output_sha256']:
            row.update(status='source_mismatch', reason='materialized file is absent or differs from audit')
        elif point['system'] == 'halo2_hw' and hashlib.sha256(compiled.get(
                point['case'].removesuffix(f"_cap{args.cap}"), '').encode()).hexdigest() != point['output_sha256']:
            row.update(status='binary_source_mismatch', reason='binary was not built from this audited source')
        else:
            function = path.name.split('.', 1)[1].rsplit('.', 1)[0]
            folder = path.parent
            fixture = folder / f'fixture.{function}.json'
            if not fixture.exists():
                fixture = folder / 'fixture.json'
            row['fixture_sha256'] = digest(fixture)
            row['tool_sha256'] = digest(args.halo2_bin) if point['system'] == 'halo2_hw' else noir_digest
            row['memory'] = args.memory
            previous = old.get(key, {})
            if previous.get('status') == 'ok' and all(previous.get(k) == v for k, v in row.items()):
                print(f"reuse {key[0]} {key[1]}", flush=True)
                continue
            work = args.work / point['case'] / point['system']
            work.mkdir(parents=True, exist_ok=True)
            output, log = work / 'counts.json', work / 'counts.log'
            output.unlink(missing_ok=True)
            if point['system'] == 'halo2_hw':
                cmd = [str(args.halo2_bin.resolve()), 'stats']
            else:
                cmd = [sys.executable, '-m', 'hw_check.noir_driver', 'setup']
            cmd += ['--case', str(folder), '--function', function, '--fixture', str(fixture),
                    '--input', 'B', '--workdir', str(work.resolve()), '--out', str(output.resolve())]
            # timeout runs INSIDE the systemd scope: killing systemd-run alone
            # would otherwise leave its workload in the separate cgroup.
            cmd = [str(args.memory_runner.resolve()), 'timeout', '--signal=TERM',
                   '--kill-after=5s', f'{args.timeout}s', *cmd]
            print(f"count {key[0]} {key[1]}", flush=True)
            # Share the original-size checker's heavy-step lock, so counts do
            # not overlap a proof/build check even across independent sessions.
            lock = Path(env.get('HW_CHECK_LOCK', '/tmp/hw_check_heavy.lock'))
            with lock.open('w') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                row.update(invoke(cmd, root, env, args.timeout + 15, log))
            row['log'] = str(log.resolve())
            if output.exists():
                row['result'] = json.loads(output.read_text())
                if not row['result'].get('ok'):
                    row['status'] = 'failed'
            elif row['status'] == 'ok':
                row.update(status='failed', reason='driver did not write its result')
        failures += row['status'] != 'ok'
        with args.out.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    return int(bool(failures))
