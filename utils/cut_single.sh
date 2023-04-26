#!/bin/bash
#SBATCH --account=rrg-istairs-ad
#SBATCH --export=NONE
#SBATCH --time=1:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=filterbank_cut
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
module use /project/6004902/modulefiles
module load presto
module load chime-psr
source ~/extract_snr/bin/activate
python ~/rrat_or_not/utils/cut_centre_filterbank.py --samples_percent $1 --input_file $2
