#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
data = np.load(sys.argv[1],allow_pickle=1)
true_mu_arr = list(float(d['true_mu']) for d in data)
true_std_arr = list(float(d['true_std']) for d in data)
true_mu_arr.sort()
true_std_arr.sort()
true_mu_arr = list(set(true_mu_arr))
true_std_arr = list(set(true_std_arr))
OR_mat = np.zeros((len(true_mu_arr),len(true_std_arr)))
for i,true_mu in enumerate(true_mu_arr):
    for j,true_std in enumerate(true_std_arr):
        for k,d in enumerate(data):
            if (true_mu==float(d['true_mu']))&(true_std==float(d['true_std'])):
                OR_mat[i,j]=d['or']
                if d['or']<0:
                    print(j)
                    print(true_mu,true_std,d['or'])
                    print(d['true_mu'],d['true_std'])
OR_mat[OR_mat<0]=-5
OR_mat[OR_mat>5]=5

plt.pcolormesh(true_mu_arr,true_std_arr,OR_mat.T)
plt.colorbar()
plt.show()
import pdb; pdb.set_trace()
