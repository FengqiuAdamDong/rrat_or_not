from utils import _theoretical_gauss_mode_exp
from utils import _find_mode_and_max

def effective_width(tau, sigma):
    mode, mode_height = _find_mode_and_max(tau, sigma)
    effective_width = 1 / mode_height
    return effective_width, mode_height


#create an array of taus with sigma = 1
import numpy as np
#logspace from 0.0001 to 130
taus = np.logspace(np.log10(1), np.log10(50), 100)
#logspace width between 0.004 and 5000
sigmas = np.logspace(np.log10(1), np.log10(50), 101)
#make a matrix of effective widths for each tau and sigma
mtau, msigma = np.meshgrid(taus, sigmas)

mew, mmh = np.zeros_like(mtau), np.zeros_like(mtau)
for i in range(msigma.shape[0]):
    for j in range(mtau.shape[1]):
        print(f"Calculating effective width for tau={mtau[i,j]:.4f}, sigma={msigma[i,j]:.4f}")
        ew, mh = effective_width(mtau[i,j], msigma[i,j])
        mew[i,j] = ew
        mmh[i,j] = mh

np.savez("effective_widths.npz", mtaus=mtau, msigmas=msigma, taus=taus, sigmas=sigmas, effective_widths=mew, mode_heights=mmh)
from matplotlib import pyplot as plt
plt.figure()
plt.pcolormesh(taus, sigmas, np.log10(mew))
plt.xscale('log')
plt.yscale('log')
plt.colorbar(label='Effective Width (ms)')
plt.xlabel('Tau (ms)')
plt.ylabel('Sigma (ms)')
plt.figure()
plt.pcolormesh(taus, sigmas, np.log10(mmh))
plt.xscale('log')
plt.yscale('log')
plt.colorbar(label='Mode Height (1/ms)')
plt.xlabel('Tau (ms)')
plt.ylabel('Sigma (ms)')
plt.show()
