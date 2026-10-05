"""Build the pinned browser/Rust dependencies, then verify both execution paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .execution import HERE, REPO, TARGET, Runner


def run(args, **kwargs):
    print('+', ' '.join(map(str,args)), flush=True)
    subprocess.run(list(map(str,args)), cwd=REPO, check=True, **kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--runtime', type=Path, default=HERE/'runtime')
    args=parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Execution currently requires Linux with bubblewrap and user namespaces.')
    for executable in ['bwrap','prlimit','rustup','cargo','npm']:
        if not shutil.which(executable):
            parser.error(f'Missing {executable}; see site/README.md for setup.')
    if not any((REPO/'zinnia').glob('_zinnia_core*.so')):
        parser.error('Build the Zinnia development extension first.')
    subprocess.run(['rustc','+risc0','--version'],check=True)
    offline=['--offline'] if args.offline else []
    run(['npm','ci','--prefix',HERE,*offline])
    run(['npm','run','build','--prefix',HERE])
    run(['cargo','+stable','build','--locked','--release','--manifest-path',HERE/'rust-host/Cargo.toml',*offline])
    env=os.environ.copy()
    env['RUSTFLAGS']='-C passes=lower-atomic -C link-arg=-Ttext=0x00200800 -C link-arg=--fatal-warnings -C panic=abort --cfg getrandom_backend="custom"'
    run(['cargo','+risc0','build','--locked','--release','--target',TARGET,'--manifest-path',HERE/'rust-guest/Cargo.toml',*offline],env=env)
    runner=Runner(args.runtime)
    status=runner.readiness()
    runner.pool.shutdown()
    print(json.dumps(status,indent=2))
    if not all(status.get(k) for k in ['sandbox','zinnia','rust']):
        raise SystemExit('Execution preflight failed. Do not start participants until both paths pass.')
    files=[HERE/'content.py',HERE/'practice_v2.py',HERE/'execution.py',HERE/'worker.py',HERE/'store.py',HERE/'server.py',HERE/'design.py',
           HERE.parent/'tasks.py',*sorted((HERE.parent/'revision2').glob('*.py')),*sorted((HERE.parent/'revision3').glob('*.py')),*sorted((HERE.parent/'revision4').glob('*.py')),*sorted((HERE.parent/'revision5').glob('*.py')),HERE/'static/manual.json',HERE/'static/requirements.json',HERE/'static/bundle.js',HERE/'static/style.css',HERE/'static/index.html',HERE/'rust-host/Cargo.lock',HERE/'rust-guest/Cargo.lock',
           HERE/'rust-host/target/release/study-risc0-host',*sorted((REPO/'zinnia').rglob('*.py')),*sorted((REPO/'zinnia').glob('*.so'))]
    manifest={'python':sys.version,'versions':{},'sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    for name,command in {'rust_host':['rustc','+stable','--version'],'rust_guest':['rustc','+risc0','--version'],'r0vm':['r0vm','--version'],'numpy':[sys.executable,'-c','import numpy; print(numpy.__version__)']}.items():
        manifest['versions'][name]=subprocess.check_output(command,text=True).strip()
    path=args.runtime/'build-manifest.json'
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Build manifest: {path}')


if __name__=='__main__':
    main()
