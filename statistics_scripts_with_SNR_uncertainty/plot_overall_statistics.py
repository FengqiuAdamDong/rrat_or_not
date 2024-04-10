import numpy as np
import matplotlib.pyplot as plt
import argparse
import smplotlib
from bayes_factor_LNLN import read_config
def get_best_fit_values(dynesty_results,calibrated_samples):
    from dynesty.utils import quantile
    if calibrated_samples is not None:
        samples = calibrated_samples
        print('Using calibrated samples')
    else:
        samples = dynesty_results.samples
    importance_weights = dynesty_results.importance_weights()
    quantiles = []
    for i in range(samples.shape[1]):
        quantiles.append(
            quantile(samples[:, i], [0.16, 0.5, 0.84], weights=importance_weights)
        )
    quantiles = np.array(quantiles)
    from dynesty.utils import mean_and_cov
    mean, cov = mean_and_cov(samples, weights=importance_weights)
    return quantiles, mean, cov


def load_data(fn):
    data = np.load(fn, allow_pickle=True)
    if "calibrated_samples" in data.keys():
        calibrated_samples = data["calibrated_samples"]
    else:
        calibrated_samples = None
    data = data['results'].tolist()
    return data, calibrated_samples


def process_npz(npz_files,yaml_files):
    means = []
    covs = []
    quantiles=[]
    N_cap = []
    for npz_file,yaml_file in zip(npz_files,yaml_files):
        print(npz_file,yaml_file)
        data, calibrated_samples = load_data(npz_file)
        quantile, mean, cov = get_best_fit_values(data,calibrated_samples)
        print(mean)
        means.append(mean)
        covs.append(np.diag(cov))
        quantiles.append(quantile)
        _,logn_N_range, _,_,_,_,_,_,_,_,_ = read_config(yaml_file)
        N_cap.append(logn_N_range[1])
    means = np.array(means)
    covs = np.array(covs)
    quantiles = np.array(quantiles)
    N_cap = np.array(N_cap)

    mu_snr = [mean[0] for mean in means]
    mu_snr_err = [cov[0] for cov in covs]
    std_snr = [mean[1] for mean in means]
    std_snr_err = [cov[1] for cov in covs]
    mu_w = [mean[2] for mean in means]
    mu_w_err = [cov[2] for cov in covs]
    std_w = [mean[3] for mean in means]
    std_w_err = [cov[3] for cov in covs]
    N = [quantile[4] for quantile in quantiles]
    mu_snr = np.array(mu_snr)
    mu_snr_err = np.array(mu_snr_err)
    std_snr = np.array(std_snr)
    std_snr_err = np.array(std_snr_err)
    mu_w = np.array(mu_w)
    mu_w_err = np.array(mu_w_err)
    std_w = np.array(std_w)
    std_w_err = np.array(std_w_err)
    N = np.array(N)




    #plot the 2d histogram of the results
    fig, ax = plt.subplots(3,2,figsize=(10,10))
    h = ax[0,0].hist2d(mu_snr,std_snr,bins=5)
    cbar = plt.colorbar(h[3], ax=ax[0,0])
    ax[0,0].set_xlabel('mu_snr')
    ax[0,0].set_ylabel('std_snr')
    h = ax[0,1].hist2d(mu_w,std_w,bins=5)
    cbar = plt.colorbar(h[3], ax=ax[0,1])
    ax[0,1].set_xlabel('mu_w')
    ax[0,1].set_ylabel('std_w')
    ax[1,0].hist(mu_snr,bins="auto")
    ax[1,0].set_xlabel('mu_snr')
    ax[1,1].hist(std_snr,bins="auto")
    ax[1,1].set_xlabel('std_snr')
    ax[2,0].hist(mu_w,bins="auto")
    ax[2,0].set_xlabel('mu_w')
    ax[2,1].hist(std_w,bins="auto")
    ax[2,1].set_xlabel('std_w')
    plt.tight_layout()
    plt.savefig('hists.png')
    fig, ax = plt.subplots(1,2,figsize=(10,10))
    print(N)
    null_all = 1-(N/N_cap[:,np.newaxis])
    # import pdb; pdb.set_trace()
    null_error = np.array([(np.abs(n[2]-n[1]),np.abs(n[0]-n[1])) for n in null_all]).T
    print(null_error)
    null = np.array([n[1] for n in null_all])
    h = ax[0].hist2d(null,mu_snr,bins=5)
    #color bar
    cbar = plt.colorbar(h[3], ax=ax[0])
    ax[0].set_xlabel('nulling fraction')
    ax[0].set_ylabel('mu_snr')
    h = ax[1].hist2d(null,mu_w,bins=5)
    cbar = plt.colorbar(h[3], ax=ax[1])
    ax[1].set_xlabel('nulling fraction')
    ax[1].set_ylabel('mu_w')
    plt.tight_layout()
    plt.savefig('hists2N.png')

    fig, ax = plt.subplots(2,2,figsize=(10,10))
    ax[0,0].errorbar(null,mu_snr,yerr=mu_snr_err,xerr=null_error,fmt='o')
    ax[0,0].set_xlabel('nulling fraction')
    ax[0,0].set_ylabel('mu_snr')
    ax[0,0].set_xlim(0,1)
    ax[0,1].errorbar(null,mu_w,yerr=mu_w_err,xerr=null_error,fmt='o')
    ax[0,1].set_xlabel('nulling fraction')
    ax[0,1].set_ylabel('mu_w')
    ax[0,1].set_xlim(0,1)
    ax[1,0].errorbar(null,std_snr,yerr=std_snr_err,xerr=null_error,fmt='o')
    ax[1,0].set_xlabel('nulling fraction')
    ax[1,0].set_ylabel('std_snr')
    ax[1,0].set_xlim(0,1)
    ax[1,1].errorbar(null,std_w,yerr=std_w_err,xerr=null_error,fmt='o')
    ax[1,1].set_xlabel('nulling fraction')
    ax[1,1].set_ylabel('std_w')
    ax[1,1].set_xlim(0,1)

    plt.tight_layout()

    plt.figure()
    plt.hist(null,bins="auto")
    plt.xlabel('Nulling fraction')


    plt.show()


if __name__=="__main__":
    import sys
    npzs = sys.argv[1:]


    npz_base = [name.split('.')[0] for name in npzs]
    yamls = [name + '.yaml' for name in npz_base]

    process_npz(npzs,yamls)
