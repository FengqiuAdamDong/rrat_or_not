import numpy as np
from read_positive_burst import read_positive_burst
from extract_snr import combine_positives
import argparse
if __name__ == "__main__":
    # fn = 'real_pulses/positive_bursts_edit_snr.csv'
    # fn1 = 'real_pulses/positive_bursts_1_edit_snr.csv'
    # fn2 = 'real_pulses/positive_bursts_short_edit_snr.csv'
    # fn = 'real_pulses/positive_burst_test.csv'
    # fn1 = fn
    # fn2 = fn
    # be sure to cutout all of the filterbank files first!!
    # dedisperse to some nominal DM to make it easier
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", default="outfn", help="output file name", required=True
    )
    parser.add_argument("-p", nargs="+", help="TOA files")
    args = parser.parse_args()

    filfiles = []
    maskfiles = []
    # in the cut out the pulse is always at 3s
    positive_fl = args.p
    fil1 = []
    dm1 = []
    toa1 = []

    for p in positive_fl:
        dm_temp, toa_temp, boxcar_det_snr, MJD, fil_temp = read_positive_burst(p)
        # fil_temp,dm_temp,toa_temp = read_positive_file(p)
        if len(fil1) == 0:
            fil1, dm1, toa1 = (fil_temp, dm_temp, toa_temp)
        else:
            fil1, dm1, toa1 = combine_positives(
                fil1, fil_temp, dm1, dm_temp, toa1, toa_temp
            )
        print(len(fil1), len(dm1), len(toa1))

    fil1 = list(set(fil1))
    for f in fil1:
        if ".fil" in f:
            filfiles.append(f)
            maskedfn = f.strip(".fil") + "_rfifind.mask"
            maskfiles.append(maskedfn)
    #fil1 has the list of files
    #maskfiles has the list of mask files
    #construct the tar command
    tarcommand = f"tar -cvf {args.o}.tar"
    for f,m in zip(filfiles,maskfiles):
        tarcommand += f" {f} {m}"
    #run the tar command
    import subprocess
    print(f"tarring {len(filfiles)} fil files and {len(maskfiles)} mask files")
    subprocess.run(tarcommand,shell=True)
