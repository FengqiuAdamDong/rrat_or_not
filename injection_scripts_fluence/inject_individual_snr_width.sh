#!/bin/bash
#SBATCH --account=rrg-istairs-ad
#SBATCH --export=NONE
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=inject_individual_snr_width
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module use /project/6004902/chimepsr-software/v1/environment-modules
module load presto
module load chime-psr
source ~/rrat_or_not_width/bin/activate
echo "python $4/inject_individual_snr_width.py --snr $1 --width $2 --dm $3"
python $4/inject_individual_snr_width.py --snr $1 --width $2 --dm $3
