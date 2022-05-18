#!/usr/bin/env python3

import numpy as np
import sys
from recover_snr import *
injections = np.load('sample_injections.npy',allow_pickle=1)
filfiles = sys.argv[1:]
#get the mask array
mask_arr = get_mask_arr(filfiles)
#grab_spectra([bfb],ts_arr,te_arr,mask_arr,dm)


import pdb; pdb.set_trace()
