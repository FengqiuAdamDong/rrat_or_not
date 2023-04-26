#!/bin/bash
#
# This script will cut the filterbank files to the desired length
for file in $@
do
    sbatch ~/rrat_or_not/utils/cut_single.sh 80 $file
done
