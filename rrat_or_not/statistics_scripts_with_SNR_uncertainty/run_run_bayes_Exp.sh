#!/bin/sh
#
for f in $1/*.dill; do
    echo "Processing $f"
    sbatch ~/rrat_or_not_with_width/rrat_or_not/statistics_scripts_with_SNR_uncertainty/run_bayes_EXPEXP.sh $f

    # python bayes_factor_Exp_no_a.py -i $f
    # take action on each file. $f store current file name
  #cat $f
done
