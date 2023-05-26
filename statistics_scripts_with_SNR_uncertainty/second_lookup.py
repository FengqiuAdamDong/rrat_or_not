import sys
import fix_second
import numpy as np
from multiprocessing import Pool
fix_second.load_detection_fn(sys.argv[1])
#create a meshgrid of mu and std from -4 to 3 and 0 to 2

mu = np.linspace(-4,3,100)
std = np.linspace(0.05,2,100)
# p = Pool(10)
second_integral = np.zeros((len(mu),len(std)))
for i in range(len(mu)):
    X = []
    for j in range(len(std)):
        X.append([mu[i],std[j]])
    # second_integral[i,:] = p.map(fix_second.second_integral,X)
    # print(f"Finished {i}th row")
    # for k in range(len(X)):
        # cpu_ = fix_second.second_integral(X[k])
        # print(cpu_)
        # gpu_ = fix_second.second_integral_cupy(X[k])
        # print(gpu_)
    for k in range(len(X)):
        second_integral[i,k] = fix_second.second_integral_cupy(X[k])
        print(f"Finished {i}th row {k}th column")
#calculate the probability of each point in the meshgrid
np.savez(f"second_integral_{sys.argv[1]}",second_integral=second_integral,mu=mu,std=std)

import matplotlib.pyplot as plt
plt.close()
plt.figure()
plt.imshow(second_integral,origin='lower',extent=[0.05,2,-4,3],aspect='auto')
plt.colorbar()
plt.xlabel("std")
plt.ylabel("mu")
plt.show()
