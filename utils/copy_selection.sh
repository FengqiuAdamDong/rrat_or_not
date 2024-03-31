#!/bin/sh
#
COPY_FILE=$1

#loop through all dir
for dir in */; do
    #check if dir is a directory
    if [ -d "$dir" ]; then
        #copy file to dir

        cp $COPY_FILE $dir/fdp/
        echo "Copied $COPY_FILE to $dir/fdp/"
    fi
done
