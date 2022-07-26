#!/usr/bin/env bash
set -euo pipefail
SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
for f in *.fil
do
    #strip the extension
    MASKFOL="${f%.*}"
    #get the maskfile
    MASK="$MASKFOL"_rfifind.mask
    echo $MASK $f
    #copy the filterbank file back in
    cp -d $f $MASKFOL
    cd $MASKFOL
    sbatch $SOURCEDIR/Inject_one_fil.sh $f $MASK
done
