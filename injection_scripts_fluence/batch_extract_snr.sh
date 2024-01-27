#!/bin/bash
#SBATCH --account=rrg-istairs-ad
#SBATCH --export=NONE
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=extract_snr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
source ~/util/load_presto.sh
source ~/rrat_or_not_width/bin/activate
ln -s */*rfifind* .
echo "python /home/adamdong/rrat_or_not_with_width/rrat_or_not/injection_scripts_fluence/extract_snr.py -dm $1 -o $2 -ds 3 -p $3 -cutout 0.8 -period $4" >> extract_command.txt
python /home/adamdong/rrat_or_not_with_width/rrat_or_not/injection_scripts_fluence/extract_snr.py -dm $1 -o $2 -ds 3 -p $3 -cutout 0.8 -period $4 >> extract_snr.log
