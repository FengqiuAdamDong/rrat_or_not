#!/bin/bash
#SBATCH --account=rrg-istairs-ad
#SBATCH --export=NONE
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=injections_det
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
LOCAL=false
while getopts a:i:l flag
do
    case "${flag}" in
        i) fil=${OPTARG};;
        l) LOCAL=true;;
        a) SCRIPT_DIR=${OPTARG};;
    esac
done
PULSAR=$(echo "$fil" | rev | cut -f2- -d '.' | rev)
EXT="${fil##*.}"

if [ "$LOCAL" != true ]; then
    module use /project/6004902/chimepsr-software/v1/environment-modules
    module load presto
    module load chime-psr
    source ~/extract_snr_py310/bin/activate
else
    SLURM_TMPDIR='/media/adam/d0fdb915-c69f-4fba-9759-ed1844c4685b/tmpdir/'$PULSAR
    echo $SLURM_TMPDIR
    mkdir -p $SLURM_TMPDIR
    SLURM_JOB_ID=1
fi
#copy the sample file
cp sample_injections.npz $SLURM_TMPDIR
#copy the filterbank files
cp $fil $SLURM_TMPDIR
#copy all the rfifind stuff so that we can find/use the masks
cp $PULSAR/*rfifind* $SLURM_TMPDIR
cd $SLURM_TMPDIR
#python "$SCRIPT_DIR"/inject_stats.py -l $fil -plot_folder fit_plots
python "$SCRIPT_DIR"/inject_stats.py -l $fil -o $PULSAR"_inj_stats.dill"
#come back
cd -
cp $SLURM_TMPDIR/$PULSAR"_inj_stats.dill" .
# cp -r $SLURM_TMPDIR/fit_plots .
