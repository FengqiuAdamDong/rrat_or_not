#!/usr/bin/env bash
set -euo pipefail
CURDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
for f in *.fil
do
    MASKFOL="${filename%.*}"
    MASK=$MASKFOL/*.mask
    sbatch $CURDIR/Inject_one_fil.sh $MASK $f
done
