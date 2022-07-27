#!/usr/bin/env bash
set -euo pipefail

SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
for FOL in *
do
    #get the maskfile
    echo $FOL
    cd $FOL
    python ~/Documents/rrat_or_not/injection_scripts/inject_stats.py -d -l positive_*.csv
    cd ..
done
