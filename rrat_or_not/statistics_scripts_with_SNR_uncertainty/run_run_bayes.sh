#!/bin/bash
for f in $1/*.dill; do
    echo $f
    sbatch ~/rrat_or_not_with_width/rrat_or_not/statistics_scripts_with_SNR_uncertainty/run_bayes_LNLN.sh $f
done
