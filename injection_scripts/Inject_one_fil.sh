#!/bin/bash
#SBATCH --account=rrg-istairs-ad
#SBATCH --export=NONE
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=injections
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

python /home/adamdong/rrat_or_not/injection_scripts/inject_pulses.py --m $2 $1
