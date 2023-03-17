#!/usr/bin/env bash
set -euo pipefail

SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
for f in *.fil
do
    #get the maskfile
    FOL="${f%.*}"
    echo $FOL
    cd $FOL
    python "$SOURCEDIR"/inject_stats.py -d -l positive_1.csv
    cd ..
done
