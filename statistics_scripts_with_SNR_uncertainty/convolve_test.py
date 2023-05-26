#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm


x = np.linspace(-10, 10, 1000)
a = norm.pdf(x, loc=5, scale=0.1)
b = norm.pdf(x, loc=-2, scale=0.1)
c = np.convolve(a, b)
x_c = np.linspace(-20, 20, 1999)
import matplotlib.pyplot as plt

plt.plot(x_c, c)
plt.show()
import pdb

pdb.set_trace()
