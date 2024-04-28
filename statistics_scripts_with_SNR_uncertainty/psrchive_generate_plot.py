import psrchive
import sys
from matplotlib import pyplot as plt
import numpy as np

pfd_file = sys.argv[1]
# load using psrchive
arch = psrchive.Archive_load(pfd_file)
total_subint = arch.get_nsubint()
total_chans = arch.get_nchan()
total_bins = arch.get_nbin()
mask_array = np.zeros((total_subint, 1, total_chans, total_bins))
for i, integration in enumerate(arch):
    for j in range(integration.get_nchan()):
        mask_array[i, 0, j, :] = integration.get_weight(j)


arch.remove_baseline()
arch.dededisperse()
data = arch.get_data()
# average over
data = np.ma.masked_array(data, mask=(mask_array == 0))
data = np.mean(data, axis=2)
data = np.mean(data, axis=1)
# keep only the middle 50% of the data
data = data[int(total_subint * 0.25) : int(total_subint * 0.75), :]
plt.imshow(data, aspect="auto", cmap="spring")
plt.show()
# plot the data
