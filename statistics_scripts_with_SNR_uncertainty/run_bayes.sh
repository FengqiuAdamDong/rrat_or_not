#!/bin/bash
#SBATCH --account=def-istairs
#SBATCH --export=NONE
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=bayes_factor_logn
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:v100l:1
#
source ~/util/load_presto.sh
source ~/extract_snr/bin/activate
python ~/rrat_or_not/statistics_scripts_with_SNR_uncertainty/bayes_factor_NS_LN_no_a_single.py -i $1
# python bayes_factor_NS_LN_no_a_single.py -i $1
