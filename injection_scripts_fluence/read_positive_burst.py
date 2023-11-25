#!/usr/bin/env python3
# THIS FUNCTION SERVES TO READ POSITIVE BURST CSV FILES
import csv


def read_positive_burst_inj(fn):
    # this function reads the positive cand files and outputs the time of detection and DM as an array
    dm = []
    time = []
    boxcar_det_snr = []
    inj_snr = []
    inj_width = []
    MJD = []
    with open(fn, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            directory = row[0]
            candidate_file_name = directory.split("/")[-1]
            filterbank_filename = directory.split("/")[-3]
            snr_field = filterbank_filename.split("_")[-2]
            width_field = filterbank_filename.split("_")[-1]
            if "SNR" in snr_field:
                inj_snr.append(float(snr_field.strip("SNR")))
                inj_width.append(float(width_field.strip("width")))
                fields = candidate_file_name.split("_")
                boxcar_det_snr.append(float(fields[-1]))
                MJD.append(float(fields[2]))
                time.append(float(fields[4]))
                dm.append(float(fields[6]))
    return dm, time, boxcar_det_snr, inj_snr, inj_width, MJD


def read_positive_burst(fn):
    # this function reads the positive cand files and outputs the time of detection and DM as an array
    dm = []
    time = []
    boxcar_det_snr = []
    MJD = []
    filterbank_filenames = []
    with open(fn, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            directory = row[0]
            candidate_file_name = directory.split("/")[-1]
            filterbank_filename = directory.split("/")[-3]
            fields = candidate_file_name.split("_")
            filterbank_filenames.append(filterbank_filename + ".fil")
            boxcar_det_snr.append(float(fields[-1]))
            MJD.append(float(fields[2]))
            time.append(float(fields[4]))
            dm.append(float(fields[6]))
    return dm, time, boxcar_det_snr, MJD, filterbank_filenames


if __name__ == "__main__":
    import sys

    dm, time, boxcar_det_snr, MJD, filterbankfn = read_positive_burst(sys.argv[1])
    dm, time, boxcar_det_snr, inj_snr, MJD = read_positive_burst_inj(
        sys.argv[1]
    )
    import pdb

    pdb.set_trace()
