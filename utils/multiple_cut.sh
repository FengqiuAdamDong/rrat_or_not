#!/bin/bash
#
# This script will cut the filterbank files to the desired length
for file in $@
do
    sbatch cut_single.sh 80 $file
done
