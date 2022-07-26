#!/usr/bin/env bash
set -euo pipefail


SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
for f in *.fil
do
    #strip the extension
    FOL="${f%.*}"
    #get the maskfile
    echo $FOL
    cd $FOL
    #copy the filterbank file back in
    /home/adamdong/CHIME-Pulsar_automated_filterbank/check_single_pulse.sh -b -d $1
    cd ..
done
