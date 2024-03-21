#!/bin/bash
# recursively go through and perform the inject_stats to get the inj_stats.dill files

source ~/util/load_presto.sh
source ~/rrat_or_not/bin/activate
AFP="$(dirname $(readlink -f $0))"
for d in */ ; do
    echo $d
    for fol in */ ; do
        cd $fol
        python $AFP/combine_multiple_inj.py *SNR*.dill
        cd ..
    done
    cd ..
done
