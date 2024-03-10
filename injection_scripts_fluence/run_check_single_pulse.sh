#!/usr/bin/env bash
set -euo pipefail
LOCAL=false
while getopts ld: flag
do
    case "${flag}" in
        l) LOCAL=true;;
        d) dm=${OPTARG};;
    esac
done
if [ "$LOCAL" != true ]; then
    SCRIPT_DIR="/home/adamdong/CHIME-Pulsar_automated_filterbank"
else
    SCRIPT_DIR="/home/adam/Documents/CHIME-Pulsar_automated_filterbank"
fi

shopt -s nullglob
for f in *.fil
do
    #strip the extension
    FOL="${f%.*}"
    #get the maskfile
    echo $FOL
    cd $FOL
    #copy the filterbank file back in
    #check how many rfifind files there are in the directory
    for inject in *SNR*.fil
    do
        #strip the extension
        inject_fol="${inject%.*}"
        #check that inject_fol exists
        if [ -d "$inject_fol" ]; then
            rfifind_files=$(find $inject_fol/ -name "*rfifind*" | wc -l)
            #check that cands.csv exists
            if [ ! -f "$inject_fol/cands.csv" ]; then
                continue
            fi
            #if less than 5 then remove the directory and continue
            if [ $rfifind_files -lt 5 ]; then
                echo "$inject_fol not enough rfifind files, removing directory"
                rm -r $inject_fol
            fi
        fi

    done

    if [ "$LOCAL" != true ]; then
        "$SCRIPT_DIR"/check_single_pulse.sh -b -d $dm *SNR*.fil
        "$SCRIPT_DIR"/check_single_pulse.sh -f *SNR*.fil
        #"$SCRIPT_DIR"/get_bright_bursts_csv_only.sh -i .
    else
        "$SCRIPT_DIR"/check_single_pulse.sh -b -d $dm -l *SNR*.fil
        "$SCRIPT_DIR"/check_single_pulse.sh -f -l *SNR*.fil
        #"$SCRIPT_DIR"/get_bright_bursts.sh -i .
    fi




    cd ..
done
