import numpy as np
import yaml
import argparse
import csv
from sigpyproc import readers as r
import os
import glob


def get_obs_time(fn, maskfn):
    # print("getting filterbank data")
    filf = r.FilReader(fn)
    hdr = filf.header
    total_time = hdr.nsamples * hdr.tsamp
    from presto import rfifind

    rfimask = rfifind.rfifind(maskfn)
    total_ints = rfimask.nint
    good_ints = len(rfimask.goodints)
    # print(good_ints / total_ints)
    good_time = total_time * good_ints / total_ints
    return good_time


# creates the yaml file for pulsar

parser = argparse.ArgumentParser(description="Create yaml file for pulsar")
parser.add_argument("csv_file", type=str, help="csv file with pulsar data")

args = parser.parse_args()
csv_file = args.csv_file

# read the csv file
pulsar_name = []
pulsar_period = []
with open(csv_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        pulsar_name.append(row[0])
        pulsar_period.append(float(row[2]))

for pulsar, period in zip(pulsar_name, pulsar_period):
    # ignore if pulsar begins with #
    if pulsar[0] == "#":
        continue
    fil_files = glob.glob(f"{pulsar}/fdp/*fdp.fil")
    obs_time = 0
    print(f"Processing {pulsar}")
    for fil_file in fil_files:
        mask_file = fil_file.replace(".fil", "_rfifind.mask")
        try:
            obs_time += get_obs_time(fil_file, mask_file)
        except Exception as e:
            # print the traceback
            import traceback

            traceback.print_exc()
            print(f"Error with {fil_file} ,probably not processed")
            # write out the fil file that is not processed into a text file
            with open("not_processed.txt", "a") as f:
                f.write(f"{fil_file}\n")
    # only use the middle 80% of the data
    obs_time = obs_time * 0.8
    N = obs_time / period
    if period < 0.5:
        width_thresh = 0.003
    else:
        width_thresh = 0.005
    # create the yaml file
    yaml_file = f"{pulsar}/fdp/{pulsar}.yaml"
    # check if yaml file already exists
    write_new_yaml = False
    if os.path.exists(yaml_file):
        # load the yaml file
        with open(yaml_file, "r") as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
            snr_thresh = yaml_dict["snr_thresh"]
            width_thresh = yaml_dict["width_thresh"]
            orig_N = yaml_dict["logn_N_range"][1]
            print(f"Original N: {orig_N}")
            print(f"New N: {N}")
            try:
                snr_upper = yaml_dict["snr_upper"]
            except KeyError:
                snr_upper = 50
            try:
                width_upper = yaml_dict["width_upper"]
            except KeyError:
                width_upper = float(28e-3)
    yaml_dict = {
        "detection_curve": "inj_stats_combine_fitted.dill",
        "logn_N_range": [-1, int(N)],
        "snr_thresh": snr_thresh,
        "width_thresh": width_thresh,
        "snr_upper": snr_upper,
        "width_upper": width_upper,
    }
    if write_new_yaml:
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_dict, f)
    #read how many lines in the filtered.csv file
    #check if the filtered.csv file exists
    if os.path.exists(f"{pulsar}/fdp/filtered.csv"):
        with open(f"{pulsar}/fdp/filtered.csv", "r") as f:
            lines = f.readlines()
            N_pulses = len(lines)
    else:
        N_pulses = 0

    # record the total number of observation hours, N max and the pulsar period
    with open("pulsar_obs_time.txt", "a") as f:
        f.write(f"{pulsar},{obs_time},{N},{period},{N_pulses}\n")
        print(f"Created {yaml_file}")
