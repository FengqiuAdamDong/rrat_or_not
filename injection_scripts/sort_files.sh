#!/usr/bin/env bash
set -euo pipefail

#this script will sort files for local analysis
SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
for f in *.fil
do
    #strip the extension
    FOL="${f%.*}"
    #get the maskfile
    mkdir -p local_analysis/$FOL
    rm -f local_analysis/$FOL/*
    cp $FOL/inj_stats.dill local_analysis/$FOL
    cp $FOL/positive*.csv local_analysis/$FOL
done
