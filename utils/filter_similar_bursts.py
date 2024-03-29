import re
import os
import sys

#arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-folder_path", help="path to folder containing the files")
parser.add_argument("-dm", help="dm of the burst",type=float)
args = parser.parse_args()
folder_path = args.folder_path
dm_arr = []
tcand_arr = []
tstart_arr = []
filename_arr = []
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        string = filename
        filename_arr.append(filename)
        tcand_match = re.search(r"_tcand_(\d+\.\d+)", string)
        if tcand_match:
            tcand = float(tcand_match.group(1))
            print(f"tcand: {tcand}")
            tcand_arr.append(tcand)

        dm_match = re.search(r"_dm_(\d+\.\d+)", string)
        if dm_match:
            dm = float(dm_match.group(1))
            print(f"dm: {dm}")
            dm_arr.append(dm)

        tstart_match = re.search(r"cand_tstart_(\d+\.\d+)", string)
        if tstart_match:
            tstart = float(tstart_match.group(1))
            print(f"tstart: {tstart}")
            tstart_arr.append(int(tstart))
#run dbscan on dm_arr tcand_arr and tstart_arr
from sklearn.cluster import DBSCAN
import numpy as np

features = np.column_stack((dm_arr, tcand_arr, tstart_arr))
errors = np.array([12, 0.2, 0.1])
features = features / errors
db = DBSCAN(eps=1, min_samples=2).fit(features)

labels = db.labels_

#find the unique labels
unique_labels = set(labels)
filename_arr = np.array(filename_arr)
dm_arr = np.array(dm_arr)
tcand_arr = np.array(tcand_arr)
tstart_arr = np.array(tstart_arr)
#include all the ones that are not in a cluster because there's no repeat there
unique_fn = filename_arr[labels == -1]
for l in unique_labels:
    print(f"l: {l}")
    if l == -1:
        continue
    indices = np.where(labels == l)[0]
    print(f"indices: {indices}")
    #figure out which file has closest dm to args.dm
    cluster_dm = dm_arr[indices]
    cluster_tstart = tstart_arr[indices]
    diff_dm = np.abs(cluster_dm - args.dm)
    min_index = np.argmin(diff_dm)
    print(f"min_index: {min_index}")
    print(f"dm: {dm_arr[indices][min_index]}")
    print(f"tcand: {tcand_arr[indices][min_index]}")
    print(f"diff_dm: {diff_dm}")
    print(f"filename: {filename_arr[indices][min_index]}")
    #show other dms in cluster
    print(f"cluster_dm: {cluster_dm}")
    unique_fn = np.append(unique_fn,filename_arr[indices][min_index])

#move the files to a new folder
if not os.path.exists("filtered"):
    os.mkdir("filtered")
import shutil
for fn in unique_fn:
    try:
        print(f"coping {fn}")
        shutil.copy(os.path.join(folder_path, fn), os.path.join("filtered", fn))
    except Exception as e:
        import traceback; print(traceback.format_exc())
        import pdb; pdb.set_trace()
#also move the csv file
try:
    shutil.copy(folder_path.replace("/",".csv"), "filtered.csv")
except Exception as e:
    shutil.copy(folder_path+".csv", "filtered.csv")
