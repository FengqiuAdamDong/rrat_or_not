#!/bin/bash
for var in "$@"
do
    python ~/Documents/CHIME-Pulsar_automated_filterbank/generate_rfi_mask.py $var &
done

