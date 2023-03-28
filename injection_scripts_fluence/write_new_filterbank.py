#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib

# matplotlib.use("pdf")
import matplotlib.pyplot as plt
import sigpyproc
from sigpyproc import utils as u
import sys


def get_filterbank_data_window(fn, duration=20):
    """
    Open the filterbank file, extract the spectra around
    the middle (+/- duration / 2) and return that data as a
    2D array.
    """
    from sigpyproc import readers as r

    print("getting filterbank data")
    filf = r.FilReader(fn)
    hdr = filf.header
    tsamp = hdr.tsamp
    fil_dur = hdr.nsamples * tsamp
    # start in the middle of the data
    start = fil_dur / 2 - duration / 2
    stop = start + duration
    print("start stop bins", start, stop)
    start_bin = int(np.round(start / tsamp))
    stop_bin = int(np.round(stop / tsamp))
    nsamp = stop_bin - start_bin
    # get the data
    _ = filf.read_block(start_bin, nsamp)
    # read the block
    ofn = fn.replace(".fil", "") + f"_dur-{int(duration)}.fil"
    _.to_file(ofn)
    return ofn


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("fil", help="Filterbank file to use as raw data background")
    parser.add_argument(
        "--d", help="Duration of required output file", type=float, default=300
    )
    parser.add_argument(
        "--CHIPSPIPE",
        help="CHIPSPIPE Directory",
        type=str,
        default="/home/adam/Documents/CHIME-Pulsar_automated_filterbank",
    )

    args = parser.parse_args()
    duration = args.d
    ifn = args.fil
    sys.path.insert(1, args.CHIPSPIPE)
    ofn = get_filterbank_data_window(ifn, duration)
    from gwg_cand_search_pipeline import run_rfifind
    from gwg_cand_search_pipeline import edit_mask

    run_rfifind(ofn.replace(".fil", ""), ".fil")
    edit_mask(ofn.replace(".fil", ""), ".fil", "_rfifind.mask")
