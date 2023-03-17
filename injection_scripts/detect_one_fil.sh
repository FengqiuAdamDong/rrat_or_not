#!/bin/bash
#SBATCH --account=rrg-istairs-ad
#SBATCH --export=NONE
#SBATCH --time=10:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=injections_det
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
while getopts i:l flag
do
    case "${flag}" in
        i) fil=${OPTARG};;
        l) LOCAL=true;;
    esac
done
PULSAR=$(echo "$fil" | rev | cut -f2- -d '.' | rev)
EXT="${fil##*.}"

if [ "$LOCAL" != true ]; then
    module use /project/6004902/modulefiles
    module load presto
    module load chime-psr
    source /home/adamdong/injections/bin/activate
else
    SLURM_TMPDIR='/media/adam/d0fdb915-c69f-4fba-9759-ed1844c4685b/tmpdir/'$PULSAR
    echo $SLURM_TMPDIR
    mkdir -p $SLURM_TMPDIR
    SLURM_JOB_ID=1
fi
#copy the sample file
cp sample_injections.npz $SLURM_TMPDIR
#copy the filterbank files
cp *snr*.fil $SLURM_TMPDIR
#copy all the rfifind stuff so that we can find/use the masks
cp */*rfifind* $SLURM_TMPDIR
cd $SLURM_TMPDIR
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
python "$SCRIPT_DIR"/inject_stats.py -l *snr*.fil
#come back
cd -
cp $SLURM_TMPDIR/inj_stats.dill .
