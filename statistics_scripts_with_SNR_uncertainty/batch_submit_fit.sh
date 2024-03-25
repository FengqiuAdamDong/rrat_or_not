#!/bin/bash
#
sbatch ~/rrat_or_not_with_width/rrat_or_not/statistics_scripts_with_SNR_uncertainty/run_bayes_LNLN.sh $1
sbatch ~/rrat_or_not_with_width/rrat_or_not/statistics_scripts_with_SNR_uncertainty/run_bayes_EXPEXP.sh $1
sbatch ~/rrat_or_not_with_width/rrat_or_not/statistics_scripts_with_SNR_uncertainty/run_bayes_LNEXP.sh $1
sbatch ~/rrat_or_not_with_width/rrat_or_not/statistics_scripts_with_SNR_uncertainty/run_bayes_EXPLN.sh $1
