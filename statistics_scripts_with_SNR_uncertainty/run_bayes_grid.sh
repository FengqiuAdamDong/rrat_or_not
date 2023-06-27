#!/bin/bash
#SBATCH --account=def-istairs
#SBATCH --export=NONE
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=Bayes_fact_real
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:v100l:1

module load python
module load scipy-stack
source ~/rrat_or_not/bin/activate
python ~/rrat_or_not/statistics_scripts_with_SNR_uncertainty/bayes_factor_real_data_LN_only.py -i $1 -c $2
