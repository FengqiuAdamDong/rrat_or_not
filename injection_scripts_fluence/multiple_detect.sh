#!/usr/bin/env bash
set -euo pipefail
LOCAL=false
while getopts l flag
do
    case "${flag}" in
        l) LOCAL=true;;
    esac
done

SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
for f in *.fil
do
    echo $f
    #strip the extension
    FOL="${f%.*}"
    #get the maskfile
    #check if inj_stats already exists
    if [ -f "inj_stats.dill" ]; then
        echo "inj_stats.dill already exists, skipping"
        cd ..
        continue
    fi

    #copy the filterbank file back in
    if [ "$LOCAL" != true ]; then
        echo "need to process" $FOL
        sbatch $SOURCEDIR/detect_one_fil.sh -i $FOL -a $SOURCEDIR
    else
        $SOURCEDIR/detect_one_fil.sh -i $FOL -l -a $SOURCEDIR
    fi
    cd ..
done
