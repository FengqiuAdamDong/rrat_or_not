import os
import sys
import subprocess
import multiprocessing
npzs = sys.argv[1:]

npz_base = [name.split('.')[0] for name in npzs]

unique_base = list(set(npz_base))
path = "/home/adam/Documents/rrat_or_not_with_width/rrat_or_not/statistics_scripts_with_SNR_uncertainty/plot_bayes_ratio.py"

def run_script(X):
    base = X[0]
    npz_files = X[1]
    yaml_file = base + ".yaml"
    subprocess.call(["python", path, '--plot_fit', yaml_file, *npz_files])

X = []
for base in unique_base:
    npz_files = [name for name in npzs if base in name]
    X.append([base, npz_files])
    # yaml_file = base + ".yaml"
    # subprocess.call(["python", path, '--plot_fit', yaml_file, *npz_files])
with multiprocessing.Pool(2) as pool:
    pool.map(run_script, X)
# for x in X:
    # run_script(x)
