#!/bin/bash
#
sbatch run_bayes_LNLN.sh $1
sbatch run_bayes_EXPEXP.sh $1
sbatch run_bayes_LNEXP.sh $1
sbatch run_bayes_EXPLN.sh $1
