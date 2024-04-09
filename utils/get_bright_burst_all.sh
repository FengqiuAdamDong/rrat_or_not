#!/bin/sh
#
for f in */; do
    cd $f/fdp/
    /home/adamdong/CHIME-Pulsar_automated_filterbank/get_bright_bursts_csv_only.sh -i .
    cd ../../
