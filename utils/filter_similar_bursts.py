import re
import os
import sys
from csv import writer
from csv import reader
from matplotlib import pyplot as plt
#arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-csv_path" ,nargs='+' , help="path to folder containing the files")
parser.add_argument("-dm", help="dm of the burst",type=float)
parser.add_argument("-copy", help="copy",action="store_true")
parser.add_argument("-period", help="period of the pulsar",type=float)
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

diff_timespacing = []
diff_timespacing_after_dbscan = []

for i,ufn in enumerate(unique_filenames):
    mask = (filename_arr == ufn)
    ufn_dm = dm_arr[mask]
    ufn_tcand = tcand_arr[mask]
    ufn_tstart = tstart_arr[mask]
    ufn_path = path_arr[mask]
    ufn_filename = filename_arr[mask]
    #include all the ones that are not in a cluster because there's no repeat there
    #correct each ufn_tcand to dm infinity
    ufn_tcand_corrected = []
    for u_dm,u_tcand in zip(ufn_dm,ufn_tcand):
        t_correction = 4.15e3 * (1/800**2)*u_dm #ms
        u_tcand = u_tcand - t_correction/1000 #s
        ufn_tcand_corrected.append(u_tcand)
    ufn_tcand_corrected = np.array(ufn_tcand_corrected)
    features = ufn_tcand_corrected.reshape(-1,1)
    errors = np.array(args.period/4)
    features = features / errors

    sorted_ufn_tcand_corrected = np.sort(ufn_tcand_corrected)
    diff_timespacing.append(np.diff(sorted_ufn_tcand_corrected))

    db = DBSCAN(eps=1, min_samples=2).fit(features)

    labels = db.labels_

    unique_labels = set(labels)
    unique_fn = ufn_filename[labels == -1]
    unique_path = ufn_path[labels == -1]
    unique_tcand = ufn_tcand[labels == -1]
    for l in unique_labels:
        # print(f"l: {l}")
        if l == -1:
            continue
        indices = (labels == l)
        #figure out which file has closest dm to args.dm
        cluster_dm = ufn_dm[indices]
        cluster_tstart = ufn_tstart[indices]
        cluster_tcand = ufn_tcand[indices]
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
        unique_tcand = np.append(unique_tcand,ufn_tcand[indices][min_index])

    sorted_unique_tcand = np.sort(unique_tcand)
    diff_timespacing_after_dbscan.append(np.diff(sorted_unique_tcand))

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
#period
diff_timespacing = np.concatenate(diff_timespacing)
diff_timespacing_after_dbscan = np.concatenate(diff_timespacing_after_dbscan)
#filter out everything above 1.6
diff_timespacing = diff_timespacing[diff_timespacing < args.period+0.5]
diff_timespacing_after_dbscan = diff_timespacing_after_dbscan[diff_timespacing_after_dbscan < args.period+0.5]
import matplotlib.pyplot as plt
import smplotlib
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].hist(diff_timespacing,bins="auto")
ax[0].set_title("Before DBSCAN filtering")
ax[0].set_xlabel("Time spacing between candidates (s)")
ax[0].set_ylabel("Frequency")
ax[1].hist(diff_timespacing_after_dbscan,bins="auto")
ax[1].set_title("After DBSCAN filtering")
ax[1].set_xlabel("Time spacing between candidates (s)")
ax[1].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("time_spacing.pdf")
plt.show()
