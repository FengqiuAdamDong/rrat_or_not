#!/bin/bash
#SBATCH --account=rrg-istairs-ad
#SBATCH --export=NONE
#SBATCH --time=10:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=injections_det
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
module use /project/6004902/modulefiles
module load presto
module load chime-psr
source /home/adamdong/injections/bin/activate
#copy the sample file
cp sample_injections.npy $SLURM_TMPDIR
#copy the filterbank files
cp *snr*.fil $SLURM_TMPDIR
#copy all the rfifind stuff so that we can find/use the masks
cp */*rfifind* $SLURM_TMPDIR
cd $SLURM_TMPDIR
python /home/adamdong/rrat_or_not/injection_scripts/inject_stats.py *snr*.fil
#come back
cd -
cp $SLURM_TMPDIR/inj_stats.dill .
