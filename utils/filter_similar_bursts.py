import re
import os
import sys
from csv import writer
from csv import reader
#arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-csv_path" ,nargs='+' , help="path to folder containing the files")
parser.add_argument("-dm", help="dm of the burst",type=float)
parser.add_argument("-copy", help="copy",type=bool,default=False)
args = parser.parse_args()
csv_path = args.csv_path
copy = args.copy
dm_arr = []
tcand_arr = []
tstart_arr = []
filename_arr = []
path_arr = []
def extract_filename_from_path(path):
    # Define the pattern to match the desired string
    splits = path.split("/")
    return splits[-3]




#read each csv file
for csv in csv_path:
    with open(csv, 'r') as read_obj:
        # Create a csv.reader object
        csv_reader = reader(read_obj, delimiter=',')
        for row in csv_reader:
            file_path = row[0]
            # if os.path.isfile(file_path+".h5"):
            string = file_path
            tcand_match = re.search(r"_tcand_(\d+\.\d+)", string)
            tcand = float(tcand_match.group(1))
            # print(f"tcand: {tcand}")
            tcand_arr.append(tcand)

            dm_match = re.search(r"_dm_(\d+\.\d+)", string)
            dm = float(dm_match.group(1))
            # print(f"dm: {dm}")
            dm_arr.append(dm)

            tstart_match = re.search(r"cand_tstart_(\d+\.\d+)", string)
            tstart = float(tstart_match.group(1))
            # print(f"tstart: {tstart}")
            tstart_arr.append(int(tstart))
            filename_arr.append(extract_filename_from_path(file_path))
            path_arr.append(file_path)



#run dbscan on dm_arr tcand_arr and tstart_arr
from sklearn.cluster import DBSCAN
import numpy as np

#find the unique labels
filename_arr = np.array(filename_arr)
path_arr = np.array(path_arr)
dm_arr = np.array(dm_arr)
tcand_arr = np.array(tcand_arr)
tstart_arr = np.array(tstart_arr)
unique_filenames = set(filename_arr)
copy_counter = 0
print(f"unique_filenames: {unique_filenames}")
for i,ufn in enumerate(unique_filenames):
    mask = (filename_arr == ufn)
    ufn_dm = dm_arr[mask]
    ufn_tcand = tcand_arr[mask]
    ufn_tstart = tstart_arr[mask]
    ufn_path = path_arr[mask]
    ufn_filename = filename_arr[mask]
    #include all the ones that are not in a cluster because there's no repeat there

    features = np.column_stack((ufn_dm, ufn_tcand))
    errors = np.array([12, 0.05])
    features = features / errors
    db = DBSCAN(eps=1, min_samples=2).fit(features)

    labels = db.labels_

    unique_labels = set(labels)

    unique_fn = ufn_filename[labels == -1]
    unique_path = ufn_path[labels == -1]
    for l in unique_labels:
        # print(f"l: {l}")
        if l == -1:
            continue
        indices = (labels == l)
        #figure out which file has closest dm to args.dm
        cluster_dm = ufn_dm[indices]
        cluster_tstart = ufn_tstart[indices]
        diff_dm = np.abs(cluster_dm - args.dm)
        min_index = np.argmin(diff_dm)

        # print(f"indices: {indices}")
        # print(f"min_index: {min_index}")
        # print(f"dm: {dm_arr[indices][min_index]}")
        # print(f"tcand: {tcand_arr[indices][min_index]}")
        # print(f"diff_dm: {diff_dm}")
        # print(f"filename: {filename_arr[indices][min_index]}")
        # #show other dms in cluster
        # print(f"cluster_dm: {cluster_dm}")
        unique_fn = np.append(unique_fn,ufn_filename[indices][min_index])
        unique_path = np.append(unique_path,ufn_path[indices][min_index])

    #move the files to a new folder
    # if not os.path.exists("filtered"):
    #     os.mkdir("filtered")
    if not os.path.exists(f"filtered_{i}"):
        os.mkdir(f"filtered_{i}")
    import shutil
    print(f"copying {len(unique_path)} files")
    if copy: 
        for fn in unique_path:
            copy_counter += 1
            fn = fn + ".png"
            # print(f"coping {fn}")
            shutil.copy(fn, f"filtered_{i}/")
        #check that all the files are there
        files = os.listdir(f"filtered_{i}")
        for fn in unique_path:
            fn = fn + ".png"
            fn = fn.split("/")[-1]
            if fn not in files:
                print(f"file {fn} not found")
                import pdb; pdb.set_trace()
    #if the filtered.csv file already exists delete it and i=0
    if i == 0:
        if os.path.exists("filtered.csv"):
            os.remove("filtered.csv")

    #write the csv file
    with open(f"filtered.csv",'a') as f:
        for fn in unique_path:
            f.write(f"{fn},1,1\n")
print(f"copy_counter: {copy_counter}")
