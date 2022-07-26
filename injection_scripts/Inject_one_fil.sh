#!/bin/bash
#SBATCH --account=rrg-istairs-ad
#SBATCH --export=NONE
#SBATCH --time=20:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=injections
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
module use /project/6004902/modulefiles
module load presto
module load chime-psr
cp $1 $SLURM_TMPDIR
MASKFOL="${1%.*}"
cp -r $MASKFOL $SLURM_TMPDIR
echo $1
echo $2
cd $SLURM_TMPDIR
python /home/adamdong/rrat_or_not/injection_scripts/inject_pulses.py --m $2 $1
#come back
cd -
cp $SLURM_TMPDIR/*snr*.fil .
