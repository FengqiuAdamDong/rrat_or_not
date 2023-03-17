#!/bin/bash
#SBATCH --account=rrg-istairs-ad
#SBATCH --export=NONE
#SBATCH --time=16:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=injections
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
while getopts m:i:l flag
do
    case "${flag}" in
        m) mask=${OPTARG};;
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
else
    SLURM_TMPDIR='/media/adam/d0fdb915-c69f-4fba-9759-ed1844c4685b/tmpdir/'$PULSAR
    echo $SLURM_TMPDIR
    mkdir -p $SLURM_TMPDIR
    SLURM_JOB_ID=1
fi
cp $fil $SLURM_TMPDIR
#we should be in the CHIPSPIPE run folder
cp -r *rfifind* $SLURM_TMPDIR
echo $fil
echo $mask
cd $SLURM_TMPDIR
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
python "$SCRIPT_DIR"/inject_pulses_sigpyproc.py --m $mask --d 50 --n 5 $fil
#come back
cd -
cp $SLURM_TMPDIR/*snr*.fil .
cp $SLURM_TMPDIR/sample_injections.npz .
# rm $fil
