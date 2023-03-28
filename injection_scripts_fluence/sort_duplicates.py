#!/usr/bin/env python3

####THE PURPOSE OF THIS SCRIPT IS TO REMOVE THE DUPLICATE PULSES DUE TO INCORRECT DEDISPERSION
import os
import sys
import numpy as np
import shutil
import os

if __name__ == "__main__":
    all_figures = os.listdir(sys.argv[1])
    nominal_dm = float(sys.argv[2])
    nominal_period = float(sys.argv[3])
    dms = []
    tsamps = []
    for fn in all_figures:
        fn = fn.replace(".png", "")
        dms.append(float(fn.split("_")[6]))
        tsamps.append(float(fn.split("_")[4]))
    dms = np.array(dms)
    tsamps = np.array(tsamps)
    sort_arg = np.argsort(tsamps)
    tsamps = tsamps[sort_arg]
    dms = dms[sort_arg]
    all_figures = np.array(all_figures)[sort_arg]
    # find difference between two pulses
    diff = np.diff(tsamps)
    moved = []
    for i, d in enumerate(diff):
        if d < (nominal_period / 4):
            dm_remove = dms[[i, i + 1]]
            dm_diff = np.abs(dm_remove - nominal_dm)
            i_remove = np.argmax(dm_diff)
            move_fig = all_figures[[i, i + 1]][i_remove]
            if not (move_fig in moved):
                shutil.copy(f"{sys.argv[1]}/{move_fig}", "scrap/")
                os.remove(f"{sys.argv[1]}/{move_fig}")
                moved.append(move_fig)
    # dm_errors = (dms> (nominal_dm+2)) | (dms<(nominal_dm-2))
    # these probably all have a counterpart
    # dm_error_times = tsamps[dm_errors]
    # for i,t in enumerate(dm_error_times):
    # second_burst =
