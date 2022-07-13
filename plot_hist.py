#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
data = np.load(sys.argv[1],allow_pickle=1)
OR = list(d['or'] for d in data)
pulses = list(len(d['det_snr']) for d in data)
print(OR)
plt.hist(OR,bins=20)
plt.figure()
plt.hist(pulses,bins=20)
plt.show()
