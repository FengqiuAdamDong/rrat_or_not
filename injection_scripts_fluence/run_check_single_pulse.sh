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
for f in *.fil
do
    #strip the extension
    FOL="${f%.*}"
    #get the maskfile
    echo $FOL
    cd $FOL
    #copy the filterbank file back in
    if [ "$LOCAL" != true ]; then
        "$SCRIPT_DIR"/check_single_pulse.sh -b -d $dm *snr*.fil
        "$SCRIPT_DIR"/check_single_pulse.sh -f *snr*.fil
        "$SCRIPT_DIR"/get_bright_bursts.sh -i .
    else
        "$SCRIPT_DIR"/check_single_pulse.sh -b -d $dm -l *snr*.fil
        "$SCRIPT_DIR"/check_single_pulse.sh -f -l *snr*.fil
        "$SCRIPT_DIR"/get_bright_bursts.sh -i .
    fi




    cd ..
done
