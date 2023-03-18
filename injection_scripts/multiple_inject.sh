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
        module use /project/6004902/modulefiles
        module load presto
        module load chime-psr
    fi
    status=$(python $SOURCEDIR/detect_inject_status.py sample_injections.npz)
    if [ "$status" != 0 ]; then
        if [ "$LOCAL" != true ]; then
            sbatch $SOURCEDIR/Inject_one_fil.sh -i $f -m $MASK -a $SOURCEDIR
        else
            $SOURCEDIR/Inject_one_fil.sh -i $f -m $MASK -l a $SOURCEDIR &
        fi
    else
        echo $MASKFOL inj complete
    fi

    cd ..
done
