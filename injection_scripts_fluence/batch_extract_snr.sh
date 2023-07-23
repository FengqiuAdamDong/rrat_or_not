#!/bin/bash
#SBATCH --account=rrg-istairs-ad
#SBATCH --export=NONE
#SBATCH --time=30:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=extract_snr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
source ~/util/load_presto.sh
source ~/extract_snr/bin/activate
python ~/rrat_or_not/injection_scripts_fluence/extract_snr.py -dm $1 -o $2 -ds 3 -p filtered_edit.csv
