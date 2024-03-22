#!/bin/bash
#
source ~/util/load_presto.sh
source ~/rrat_or_not/bin/activate
AFP="$(dirname $(readlink -f $0))"
for d in */ ; do
    echo $d
    cd $d
    python $AFP/combine_inject_stats.py *.fil
    cd ..
done
