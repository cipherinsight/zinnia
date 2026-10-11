import pathlib,json,time,subprocess,shutil,run
B=pathlib.Path(__file__).resolve().parent;d=B/'pythran_fannkuch'
# Let the measured bounded candidate finish, then preserve it as diagnostic history.
while any('/examples/circ --language zsharp '+str(d/'rewrite.zok') in row for row in subprocess.check_output(['ps','-eo','args'],text=True).splitlines()):time.sleep(1)
time.sleep(1)
for n in ['result.json','rewrite.zok','rewrite.compile.log','rewrite.command.json']:
 if (d/n).exists():shutil.copyfile(d/n,d/('bounded.'+n))
(d/'rewrite.zok').write_text('''// The entry returns None; the pure helper result is discarded.
// n=0 raises IndexError inside the original helper. Keep it invalid.
def main(private u64 n) -> u64[0]:
    assert(n > 0u64 && n <= 1024u64)
    return [0u64; 0]
''')
(d/'translation.json').write_text(json.dumps({'mode':'rewrite','rewrites':['Delete the pure, unused fannkuch(n) call in fannkuch2 (kernel.py:44). It returns a value discarded by the entry and mutates only locally allocated lists; the observable result is None for every successful input.','Represent None by an empty public output and preserve the original n=0 IndexError as an assertion failure. For 1<=n<=1024 the permutation enumeration terminates and has no externally visible effects.'],'notes':'This rewrite removes no returned output: the declared entry is fannkuch2, not the helper fannkuch. Both full bounded native translations and their resource diagnostics are preserved under ungrouped.* and bounded.*.'},indent=2))
run.process(d)
