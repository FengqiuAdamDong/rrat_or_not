#!/bin/bash
#
# This script will cut the filterbank files to the desired length
for file in $@
do
    #strip the .fil extension
    file=${file%.fil}
    #check if the file is already cut
    if [ -f ${file}_cut_80.fil ]
    then
        echo "File ${file}_cut.fil already exists, skipping"
        continue
    else
        sbatch ~/rrat_or_not/utils/cut_single.sh 80 $file
    fi
done
