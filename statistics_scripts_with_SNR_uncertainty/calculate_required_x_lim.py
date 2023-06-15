import numpy as np
det_error = 0.4
def lognorm_dist(x, mu, sigma, a=0):
    #a is the shift parameter
    pdf = np.exp(-((np.log(x-a) - mu) ** 2) / (2 * sigma**2)) / (
        (x-a) * sigma * np.sqrt(2 * np.pi)
    )
    return pdf
def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

def second(n,mu,std,N,xlim=2,a=0):
     #xlim needs to be at least as large as 5 sigma_snrs though
    wide_enough = False
    sigma_snr = det_error
    maximum_accuracy = 1/(N-n)
    while not wide_enough:
        x_lims = [-sigma_snr*10,xlim]
        # x_lims = [-xlim,xlim]
        LN_dist = lognorm_dist(xlim,mu,std,a=0)
        gaussian_error = gaussian(xlim,0,sigma_snr)
        if (gaussian_error < maximum_accuracy)&(LN_dist < maximum_accuracy):
            wide_enough = True
        else:
            xlim = xlim+0.1
    return xlim

def second_wrapper(X):
    n=200
    mu=X[0]
    std=X[1]
    N=X[2]
    return second(n,mu,std,N)

def first(mu,std,xlim=0.1,x_len=10000,a=0):
    #xlim needs to be at least as large as 5 sigma_snrs though
    sigma_snr = det_error
    wide_enough = False
    while not wide_enough:
        x_lims = [-xlim,xlim]
        amp_arr = np.linspace(x_lims[0],x_lims[1],x_len)
        LN_dist = lognorm_dist(amp_arr,mu,std,a=0)
        gaussian_error = gaussian(amp_arr,0,sigma_snr)
        if (gaussian_error[-1] < 1e-5)&(LN_dist[-1] < 1e-5)&(xlim>(max(amp)+10)):
            wide_enough = True
        else:
            xlim = xlim+0.1
    return xlim
def first_wrapper(X):
    mu=X[0]
    std=X[1]
    return first(mu,std)

mu = np.linspace(-20,2,100)
std = np.linspace(0.01,4,101)
N = np.linspace(1000,10e6,102)
xlim_second = np.zeros((len(mu),len(std),len(N)))
from multiprocessing import Pool
with Pool(50) as p:
    for i,m in enumerate(mu):
        for j,s in enumerate(std):
            X = []
            for n in N:
                X.append([m,s,n])
            xlim_second[i,j,:] = p.map(second_wrapper,X)
            print(i,j)
np.savez('xlim_second_lookup.npz',xlim_second=xlim_second,mu=mu,std=std,N=N)
import pdb; pdb.set_trace()
