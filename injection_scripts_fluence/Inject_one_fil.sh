#!/bin/bash
#SBATCH --account=rrg-istairs-ad
#SBATCH --export=NONE
#SBATCH --time=3:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=injections
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
LOCAL=false
while getopts a:m:i:l flag
do
    case "${flag}" in
        m) mask=${OPTARG};;
        i) fil=${OPTARG};;
        l) LOCAL=true;;
        a) SCRIPT_DIR=${OPTARG};;
    esac
done
echo $SCRIPT_DIR
PULSAR=$(echo "$fil" | rev | cut -f2- -d '.' | rev)
EXT="${fil##*.}"

if [ "$LOCAL" != true ]; then
    module use /project/6004902/modulefiles
    module load presto
    module load chime-psr
    module load psrchive
    source ~/extract_snr/bin/activate
else
    SLURM_TMPDIR='/media/adam/d0fdb915-c69f-4fba-9759-ed1844c4685b/tmpdir/'$PULSAR
    echo $SLURM_TMPDIR
    mkdir -p $SLURM_TMPDIR
    SLURM_JOB_ID=1
fi
cp $fil $SLURM_TMPDIR
#we should be in the CHIPSPIPE run folder
cp -r *rfifind* $SLURM_TMPDIR
cp *.pazi $SLURM_TMPDIR
echo $fil
echo $mask
cd $SLURM_TMPDIR
python "$SCRIPT_DIR"/inject_pulses_sigpyproc.py --m $mask --d 400 --n 50 $fil
#come back
cd -
cp $SLURM_TMPDIR/*SNR*.fil .
cp $SLURM_TMPDIR/sample_injections.npz .
#clean up tmpdir
rm -r $SLURM_TMPDIR
# rm $fil
