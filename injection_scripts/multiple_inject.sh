#!/usr/bin/env bash
set -euo pipefail
SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOCAL=false
while getopts l flag
do
    case "${flag}" in
        l) LOCAL=true;;
    esac
done

for f in *.fil
do
    #strip the extension
    MASKFOL="${f%.*}"
    #get the maskfile
    MASK="$MASKFOL"_rfifind.mask
    echo $MASK $f
    #copy the filterbank file back in
    cp -d $f $MASKFOL
    cd $MASKFOL
    if [ "$LOCAL" != true ]; then
        sbatch $SOURCEDIR/Inject_one_fil.sh -i $f -m $MASK
    else
        $SOURCEDIR/Inject_one_fil.sh -i $f -m $MASK -l &
    fi

    cd ..
done
