#!/usr/bin/env python3
from sigpyproc import readers as r
import sys
import os
#THIS SCRIPT GETS THE TOTAL OBSERVATION TIME FOR LOTS OF FILTERBANK FILES
def get_obs_time(fn,maskfn):
    print("getting filterbank data")
    filf = r.FilReader(fn)
    hdr = filf.header
    total_time = hdr.nsamples*hdr.tsamp
    from presto import rfifind
    rfimask = rfifind.rfifind(maskfn)
    total_ints = rfimask.nint
    good_ints = len(rfimask.goodints)
    print(good_ints/total_ints)
    good_time = total_time*good_ints/total_ints
    return good_time

if __name__=="__main__":
    fns = sys.argv[1:]
    total_obs_time = 0
    for fn in fns:
        mask_name = fn.replace(".fil","_rfifind.mask")
        total_obs_time += get_obs_time(fn,mask_name)
    print(total_obs_time)
