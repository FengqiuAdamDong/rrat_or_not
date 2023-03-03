#!/usr/bin/env python3
#THIS FUNCTION SERVES TO READ POSITIVE BURST CSV FILES
import csv

def read_positive_burst(fn):
    #this function reads the positive cand files and outputs the time of detection and DM as an array
    dm = []
    time = []
    boxcar_det_snr = []
    inj_snr = []
    MJD = []
    with open(fn,"r") as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            directory = row[0]
            candidate_file_name = directory.split('/')[-1]
            filterbank_filename = directory.split('/')[-3]
            snr_field = filterbank_filename.split('_')[-1]
            inj_snr.append(float(snr_field.strip('snr')))
            fields = candidate_file_name.split('_')
            boxcar_det_snr.append(float(fields[-1]))
            MJD.append(float(fields[2]))
            time.append(float(fields[4]))
            dm.append(float(fields[6]))
    return dm,time,boxcar_det_snr,inj_snr,MJD

if __name__=="__main__":
    import sys
    dm,time,boxcar_det_snr,MJD = read_positive_burst(sys.argv[1])
    import pdb; pdb.set_trace()
