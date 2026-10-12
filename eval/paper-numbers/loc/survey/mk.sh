# mk.sh NAME DROP_FNS [EXTRA_SH]: privatized crate without phase0/1 and legacy halo2, run check, count dead code
set -e
N=$1; FNS=$2
python3 privatize.py work/nontest/src $N.src "phase0,phase1" "analyze_shapes,build_candidates,$FNS"
rm -rf $N && cp -r crate0 $N && rm -rf $N/src && mv $N.src $N/src
rm -rf $N/src/prove/halo2 && sed -i 's/^#\[cfg(feature = "halo2-backend")\]$//; s/^pub(crate) mod halo2;$//' $N/src/prove/mod.rs
[ -n "$3" ] && (cd $N && eval "$3")
source ./env.sh
(cd $N && cargo check --features halo2-base-backend,noir-backend --message-format=json 2>/dev/null > ../$N.json) || true
python3 -c "
import json,sys
for l in open('$N.json'):
  m=json.loads(l)
  if m.get('reason')=='compiler-message' and m['message']['level']=='error': print(m['message']['rendered'][:400])" | head -40
python3 deadlines.py $N.json $N > $N.dead.txt; head -1 $N.dead.txt
