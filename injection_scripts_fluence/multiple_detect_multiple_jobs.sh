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
    cd $FOL
    echo $PWD
    if [ -f "inj_stats.dill" ]; then
        echo "inj_stats.dill already exists, skipping"
        cd ..
        continue
    fi
    for inj in *SNR*.fil
    do
        PULSAR=$(echo "$inj" | rev | cut -f2- -d '.' | rev)
        inj_stats_name="$PULSAR"_inj_stats.dill
        if [ -f "$inj_stats_name" ]; then
            echo "$inj_stats_name already exists, skipping"
            continue
        fi
        #copy the filterbank file back in
        if [ "$LOCAL" != true ]; then
            echo "need to process" $inj
            sbatch $SOURCEDIR/one_snr_width_det.sh -i $inj -a $SOURCEDIR
        else
            $SOURCEDIR/one_snr_width_det.sh -i $inj -l -a $SOURCEDIR &
        fi
    done

    cd ..
done
