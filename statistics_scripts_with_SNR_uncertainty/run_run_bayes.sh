#!/bin/bash
for f in simulated_dir_2000/*.dill; do
    echo $f
    sbatch run_bayes.sh -i $f
done
