#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

data = np.load(sys.argv[1],allow_pickle=1)
k_arr = list(float(d['true_k']) for d in data)
or_arr = list(float(d['or']) for d in data)

# OR_mat[OR_mat<0]=-5
# OR_mat[OR_mat>5]=5
plt.figure()
plt.scatter(k_arr,or_arr,marker='.',s=3)
plt.hlines(0,0,1,colors='red')
plt.show()
import pdb; pdb.set_trace()
