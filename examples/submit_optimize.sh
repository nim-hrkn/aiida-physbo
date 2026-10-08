#!/bin/bash
# Closed-loop Bayesian optimization in the AiiDA daemon (PhysboOptimizeWorkChain) on a PHYSBO test function,
# on a discrete grid or on the continuous box.
#
#   examples/submit_optimize.sh discrete [test_function] [n_random] [n_bayes]
#   examples/submit_optimize.sh range    [test_function] [n_random] [n_bayes]
#
# The daemon must be running (`verdi daemon status`). The script waits up to ~3 minutes and prints the summary.
set -euo pipefail

SPACE=${1:-discrete}
FN=${2:-Sphere}
N_RANDOM=${3:-5}
N_BAYES=${4:-10}

J() { physbo-aiida --json "$@" 2>/dev/null; }
field() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

RANGE_OPTS=""
[ "$SPACE" = range ] && RANGE_OPTS="--optimizer-nsamples 500"     # optimizer options exist for a range space only
S=$(J submit-optimize --test-function "$FN" --space "$SPACE" --num 21 --num-random "$N_RANDOM" --num-bayes "$N_BAYES" \
      --score EI --seed 7 $RANGE_OPTS --label "example_opt_${SPACE}")
PK=$(echo "$S" | field "d['pk']")
echo "submitted PhysboOptimizeWorkChain pk $PK on a $SPACE space ($(echo "$S" | field "d['daemon']"))"

for i in $(seq 1 5); do
  W=$(J wait --pk "$PK" --wait-seconds 40)
  echo "  $(echo "$W" | field "'%s (exit %s)' % (d['state'], d['exit_status'])")"
  echo "$W" | grep -q '"terminated": true' && break
done

J results --pk "$PK" | field "'steps %s, best %s' % (d.get('num_steps_done'), (d.get('summary') or {}).get('best'))"
echo "physbo-aiida results --pk $PK ; physbo-aiida history --pk $PK --minimize ; physbo-aiida plot --pk $PK --minimize"
