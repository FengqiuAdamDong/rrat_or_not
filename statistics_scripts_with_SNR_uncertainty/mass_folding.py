import numpy as np
import os
import sys
import argparse
from csv import reader
import multiprocessing as mp
import shutil
import glob
import subprocess
from sigpyproc.readers import FilReader
from matplotlib import pyplot as plt


def run_prepfold(X):
    filterbank_file, dm, period = X
    # get the rfi mask file
    r = FilReader(filterbank_file)
    tsamp = r.header.tsamp
    nsamp = r.header.nsamples
    total_time = tsamp * nsamp
    # number of folds is npart
    folds = int(total_time / period)
    npart = int(folds / 4)

    rfi_mask = filterbank_file.replace(".fil", "_rfifind.mask")
    ignored_channels = "1023,989,988,987,986,985,984,983,982,981,980,979,978,977,976,910,909,908,907,906,905,904,903,902,901,900,899,898,897,896,895,894,893,892,891,890,889,888,887,886,885,884,883,882,881,880,879,878,877,876,875,874,873,872,871,870,869,868,867,866,865,864,863,862,861,860,859,858,857,856,855,854,853,852,851,850,849,848,847,846,845,838,837,836,835,834,833,832,831,830,829,828,827,826,825,824,823,822,821,820,819,818,817,816,815,814,813,805,804,803,802,801,800,799,798,797,796,795,794,793,792,791,790,789,788,787,786,785,784,783,782,781,780,779,778,777,776,775,774,773,772,771,770,769,618,617,616,615,614,613,612,611,610,609,608,607,606,605,604,603,602,601,600,599,598,597,596,595,594,593,592,591,590,589,588,587,586,585,584,583,582,581,580,579,578,577,576,575,574,573,572,571,570,569,568,567,566,565,564,563,562,561,560,559,558,557,556,555,554,471,470,469,468,467,466,465,464,463,462,461,460,459,458,457,456,455,451,448,439,438,437,436,435,434,433,432,431,430,429,428,427,426,392,391,390,389,388,387,386,385,384,383,382,381,380,379,346,345,344,343,342,341,340,339,338,337,336,335,334,333,332,331,330,269,268,267,266,265,264,263,262,261,235,234,233,232,177,169,168,167,166,165,164,163,150,149,148,140,136"
    base_name = filterbank_file.replace(".fil", "")
    command = f"prepfold -noxwin -nosearch -nodmsearch -dm {dm} -p {period} -ignorechan {ignored_channels} -mask {rfi_mask} -o {base_name} -npart {npart} {filterbank_file}"
    subprocess.run(command, shell=True)
    # get the pfd generated
    pfd_file = glob.glob(f"{base_name}*.pfd")[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="folding all filterbank files given a csv file of pulsars"
    )
    parser.add_argument("csv_file", type=str, help="csv file containing pulsar names")

    args = parser.parse_args()
    csv_file = args.csv_file
    with open(csv_file, "r") as read_obj:
        csv_reader = reader(read_obj, delimiter=",")
        pulsar_name = []
        dm = []
        period = []
        for row in csv_reader:
            if row[0][0] == "#":
                continue
            pulsar_name.append(row[0])
            dm.append(row[1])
            period.append(float(row[2]))

    for pulsar, d, p in zip(pulsar_name, dm, period):
        path = f"{pulsar}/fdp/"
        os.chdir(path)
        # get all the filterbank files
        filterbank_files = glob.glob("*.fil")
        X = []
        #fold 4 filterbank files per pulsar
        filterbank_files = filterbank_files[:4]
        for filterbank_file in filterbank_files:
            X.append((filterbank_file, d, p))
        # for x in X:
            # run_prepfold(x)
        with mp.Pool(4) as pool:
            pool.map(run_prepfold, X)
        os.chdir("../../")
