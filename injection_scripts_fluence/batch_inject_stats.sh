#!/bin/bash
# recursively go through and perform the inject_stats to get the inj_stats.dill files

source ~/util/load_presto.sh
source ~/rrat_or_not/bin/activate
AFP="$(dirname $(readlink -f $0))"
for d in */ ; do
    echo $d
    for fol in */ ; do
        #check if there are any *SNR*.dill files in the folder
        if [ ! -f $fol/*SNR*.dill ]; then
            continue
        fi
        cd $fol
        python $AFP/combine_multiple_inj.py *SNR*.dill
        cd ..
    done
    cd ..
done
