#!/usr/bin/env bash
set -euo pipefail

SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
for f in *.fil
do
    #strip the extension
    FOL="${f%.*}"
    #get the maskfile
    echo $FOL
    #copy the filterbank file back in
    sbatch $SOURCEDIR/detect_one_fil.sh *snr*.fil
done
