#!/bin/bash
# Interactive Bayesian optimization with the physbo-aiida CLI, on a discrete or a range space.
#
#   examples/interactive_loop.sh discrete [n_random] [n_bayes]
#   examples/interactive_loop.sh range    [n_random] [n_bayes]
#
# The objective f(x, y) = (x - 0.5)^2 + (y + 1)^2 is minimized (true minimum 0 at (0.5, -1)); it stands
# in for an experiment or a calculation evaluated outside AiiDA. Every step (space, proposals,
# observations) becomes a node in the AiiDA profile; the final observations pk is printed at the end.
# Requires physbo-aiida on PATH (the conda env that holds the AiiDA profile) and python3.
set -euo pipefail

SPACE=${1:-discrete}
N_RANDOM=${2:-5}
N_BAYES=${3:-5}
LABEL="example_${SPACE}"

J() { physbo-aiida --json "$@" 2>/dev/null; }
field() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }
objective() {   # X as JSON rows -> comma separated f values
  python3 -c "import json,sys; X=json.loads(sys.argv[1]); print(','.join(str((x-0.5)**2+(y+1)**2) for x,y in X))" "$1"
}

if [ "$SPACE" = discrete ]; then
  SPK=$(J candidates --grid '{"min":[-2,-2],"max":[2,2],"num":21}' --names x,y --label "$LABEL" | field "d['pk']")
  echo "CandidatesData pk $SPK (441 candidates)"
else
  SPK=$(J search-box --min -2,-2 --max 2,2 --names x,y --label "$LABEL" | field "d['pk']")
  echo "SearchBoxData pk $SPK (box [-2,2]^2)"
fi

OBS=""
for step in $(seq 0 $N_BAYES); do
  if [ "$step" -eq 0 ]; then
    P=$(J propose --space-pk "$SPK" --num-search-each-probe "$N_RANDOM" --seed 42 --label "$LABEL")
  else
    P=$(J propose --space-pk "$SPK" --observations-pk "$OBS" --score EI --minimize --seed "$step" --label "$LABEL")
  fi
  X=$(echo "$P" | field "json.dumps(d['X'])")
  VALUES=$(objective "$X")
  if [ "$SPACE" = discrete ]; then
    ACTIONS=$(echo "$P" | field "','.join(map(str, d['actions']))")
    O=$(J observe --space-pk "$SPK" ${OBS:+--observations-pk "$OBS"} --actions "$ACTIONS" --values "$VALUES" --label "$LABEL")
  else
    O=$(J observe --space-pk "$SPK" ${OBS:+--observations-pk "$OBS"} --x "$X" --values "$VALUES" --label "$LABEL")
  fi
  OBS=$(echo "$O" | field "d['pk']")
  echo "step $step ($(echo "$P" | field "d['summary']['mode']")): X=$X f=$VALUES -> observations pk $OBS"
done

echo
J history --pk "$OBS" --minimize | field "'best: f=%s at X=%s after %d observations' % (d['best']['best_value'], d['best']['best_X'], d['num_observations'])"
J plot --pk "$OBS" --minimize | field "'figures: ' + ', '.join(d['files'])"
echo "observations pk $OBS  (physbo-aiida history --pk $OBS --minimize)"
