#!/bin/bash
#SBATCH --account=def-istairs
#SBATCH --export=NONE
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=bayes_factor_LNExp
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:v100l:1
#
source ~/util/load_presto.sh
source ~/rrat_or_not_width/bin/activate
python ~/rrat_or_not/statistics_scripts_with_SNR_uncertainty/bayes_factor_EXPLN.py -i $1
