#!/bin/bash
#
for f in $1/*.dill; do
    echo $f
    yaml=${f%.dill}.yaml
    echo $yaml
    sbatch ~/rrat_or_not/statistics_scripts_with_SNR_uncertainty/run_bayes_grid.sh $f $yaml

done
