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
    #strip the extension
    FOL="${f%.*}"
    #get the maskfile
    echo $FOL
    cd $FOL
    #copy the filterbank file back in
    if [ "$LOCAL" != true ]; then
        sbatch $SOURCEDIR/detect_one_fil.sh -i $FOL -a $SOURCEDIR
    else
        $SOURCEDIR/detect_one_fil.sh -i $FOL -l -a $SOURCEDIR &
    fi
    cd ..
done
