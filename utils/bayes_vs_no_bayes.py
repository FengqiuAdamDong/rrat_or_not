#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import dynesty
import sys
from bayes_factor_NS_LN_no_a_single import loglikelihood
from bayes_factor_NS_LN_no_a_single import pt_Uniform_N
import yaml
import smplotlib
import statistics_basic
import dill
def process_detection_results(real_det):
    with open(real_det, "rb") as inf:
        det_class = dill.load(inf)

    det_fluence = []
    det_width = []
    det_snr = []
    noise_std = []

    for pulse_obj in det_class.sorted_pulses:
        if pulse_obj.det_amp != -1:
            det_fluence.append(pulse_obj.det_fluence)
            det_width.append(pulse_obj.det_std)
            det_snr.append(pulse_obj.det_snr)
            noise_std.append(pulse_obj.noise_std)

    det_fluence = np.array(det_fluence)
    det_width = np.array(det_width)
    det_snr = np.array(det_snr)
    noise_std = np.array(noise_std)

    return det_fluence, det_width, det_snr, noise_std

class dynesty_plot:
    """
    Class to plot dynesty results"""

    def __init__(self, filenames):
        self.filename = filenames
        #strip the .h5 from the filename and all .dill
        #remove the last element of the filename
        splits = [f.split('_')[:-1] for f in self.filename]
        self.detections = ['_'.join(s)+'.dill' for s in splits]
        self.yaml_files = ['_'.join(s)+'.yaml' for s in splits]
    def load_filenames(self):
        """
        Load the filenames into a list of data objects
        """
        self.data = []
        for f in self.filename:
            if f.endswith('.npz'):
                self.data.append(np.load(f,allow_pickle=True)['results'])
            elif f.endswith('.h5'):
                print(f)
                self.data.append(dynesty.NestedSampler.restore(f))

    def load_detections(self, num_bins=100):
        self.snrs = []
        self.snr_bin_midpoints = []
        self.snr_bin_heights = []
        self.simp_total_N = []
        self.simp_selection_corrected_heights = []
        for f in self.detections:
            det_fluence, det_width, det_snr, noise_std = process_detection_results(f)
            # Define the number of bins and the range of values
            value_min = np.min(det_snr)
            value_max = np.max(det_snr)
            # Create the bins and calculate the bin heights
            bin_heights, bin_edges = np.histogram(det_snr, bins=num_bins, range=(value_min, value_max))
            bin_midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            self.snr_bin_midpoints.append(bin_midpoints)
            self.snr_bin_heights.append(bin_heights)
            self.snrs.append(det_snr)

            simp_selection_corrected_height = bin_heights/statistics_basic.p_detect_cpu(bin_midpoints)
            self.simp_selection_corrected_heights.append(simp_selection_corrected_height)
            #cut off values after snr_bin_midpoints > 1.6
            # plt.figure()
            # plt.plot(bin_midpoints, simp_selection_corrected_height, label="corrected")
            # plt.plot(bin_midpoints, bin_heights, label="No correction")
            # plt.legend()
            # plt.show()
            self.simp_total_N.append(np.sum(simp_selection_corrected_height))
            self.simp_total_N_err = np.sqrt(self.simp_total_N)


    def plot_corner(self, labels=None, plot=False):
        """
        Plot the corner plot of the samples
        """
        from dynesty import plotting as dyplot
        means = []
        stds = []
        for f,d in zip(self.filename,self.data):
            if plot:
                corner = dyplot.cornerplot(d.results, labels=labels)
                if (period > 0) & (obs_time > 0):
                    #if these two global variables are set
                    N_ax = corner[1][2][2]
                    N_sax = N_ax.secondary_xaxis('top', functions=(Ntonull,nulltoN))
                    N_sax.set_xlabel(r"Nulling fraction")
                plt.savefig(f.strip(".h5")+"_corner.png")
                plt.close()

            mean,cov = dynesty.utils.mean_and_cov(d.results.samples,d.results.importance_weights())
            diag_cov = np.diag(cov)
            std = np.sqrt(diag_cov)
            means.append(mean)
            stds.append(std)
        self.means = np.array(means)
        self.stds = np.array(stds)

    def plot_accuracy_logn(self):
        """
        Plot the accuracy of the results
        """
        true_centres = []
        for fn,centre,errors,yaml_file in zip(self.filename,self.means,self.stds,self.yaml_files):
            #get the mu and sigma from the filename
            with open(yaml_file) as f:
                yaml_data = yaml.safe_load(f)
                true_centres.append(np.array([yaml_data['mu'],yaml_data['std'],yaml_data['N']]))

        #plot the first element of the ratios
        true_mus = [r[0] for r in true_centres]
        mus = [r[0] for r in self.means]
        mu_errs = [r[0] for r in self.stds]
        true_sigmas = [r[1] for r in true_centres]
        sigmas = [r[1] for r in self.means]
        sigma_errs = [r[1] for r in self.stds]
        true_Ns = [r[2] for r in true_centres]
        Ns = [r[2] for r in self.means]
        N_errs = [r[2] for r in self.stds]
        fitted_mu = []
        fitted_mu_err = []
        fitted_sigma = []
        fitted_sigma_err = []
        fitted_Ns = []
        #estimate k for each set of bins
        from scipy.optimize import curve_fit
        import scipy

        for i,(bin_centers,corrected_heights) in enumerate(zip(self.snr_bin_midpoints,self.simp_selection_corrected_heights)):
            #fit an exponential to the data
            def lognorm_dist(x, A, mu, sigma, lower_c=0, upper_c=np.inf):
                #lower and upper cutoff parameters added
                pdf = np.zeros(x.shape)
                mask = (x > lower_c) & (x < upper_c)
                pdf[mask] = np.exp(-((np.log(x[mask]) - mu) ** 2) / (2 * sigma**2)) / (
                    (x[mask]) * sigma * np.sqrt(2 * np.pi)
                )
                def argument(c,mu,sigma):
                    if c==0:
                        return -np.inf
                    return (np.log(c)-mu)/(sigma*np.sqrt(2))
                pdf = 2*pdf / (scipy.special.erf(argument(upper_c,mu,sigma))-scipy.special.erf(argument(lower_c,mu,sigma)))
                pdf = A*pdf
                return pdf
            corrected_heights = np.array(corrected_heights)
            bin_centers = np.array(bin_centers)
            corrected_heights = corrected_heights[bin_centers>2]
            bin_centers = bin_centers[bin_centers>2]
            popt,pcov = curve_fit(lognorm_dist,bin_centers,corrected_heights,p0=[100,0,9.5],bounds=([0,-4,0.001],[np.inf,5,10]),maxfev=1000000)
            #plot the fits
            plt.figure()
            plt.scatter(bin_centers,corrected_heights)
            plt.plot(bin_centers,lognorm_dist(bin_centers,*popt))
            plt.savefig(f"lognorm_fit_debug_{i}.png")
            plt.close()

            if popt[1] > -2:
                fitted_mu.append(popt[1])
                fitted_mu_err.append(np.sqrt(np.diag(pcov))[1])
                fitted_sigma.append(popt[2])
                fitted_sigma_err.append(np.sqrt(np.diag(pcov))[2])
                #integrate under the fitted k line
                print(popt)
                print(np.sqrt(np.diag(pcov)))
            else:
                fitted_mu.append(np.nan)
                fitted_mu_err.append(np.nan)
                fitted_sigma.append(np.nan)
                fitted_sigma_err.append(np.nan)



        plt.figure()
        max_mu = max([max(true_mus),max(mus)])
        min_mu = min([min(true_mus),min(mus)])
        plt.errorbar(true_mus,mus,yerr=mu_errs,label=r"LuNfit",linestyle='None',marker='o')
        plt.errorbar(true_mus,fitted_mu,yerr=fitted_mu_err,label=r"Simple Correction",linestyle='None',marker='x')
        x = np.linspace(min_mu,max_mu,100)
        plt.plot(x,x,'r--')
        plt.xlabel(r"True $\mu$")
        plt.ylabel(r"recovered $\mu$")
        plt.legend()
        plt.savefig('mus.png')
        plt.savefig('mus.pdf')
        plt.close()
        plt.figure()
        sigmas = np.array(sigmas)
        true_sigmas = np.array(true_sigmas)
        fitted_sigma = np.array(fitted_sigma)
        fitted_sigma_err = np.array(fitted_sigma_err)
        plt.errorbar(range(len(sigmas)),sigmas/true_sigmas,yerr=sigma_errs/true_sigmas,label=r"LuNfit",linestyle='None',marker='o')
        plt.errorbar(range(len(sigmas)),fitted_sigma/true_sigmas,yerr=fitted_sigma_err/true_sigmas,label=r"Simple Correction",linestyle='None',marker='x')
        max_sigma = max([max(true_sigmas),max(sigmas)])
        x = np.linspace(0,max_sigma,100)
        plt.axhline(1,linestyle='--',color='r')
        plt.ylabel(r"recovered $\sigma$/true $\sigma$")
        plt.xlabel(r"Index")
        plt.legend()
        plt.savefig('sigmas.png')
        plt.savefig('sigmas.pdf')
        plt.close()
        plt.figure()
        max_N = max([max(true_Ns),max(Ns)])
        x = np.linspace(0,max_N,100)
        plt.errorbar(true_Ns,Ns,yerr=N_errs,label="LuNfit",linestyle='None',marker='o')
        plt.errorbar(true_Ns,self.simp_total_N,label="Simple Correction",linestyle='None',marker='x')
        plt.xlabel("True N")
        plt.ylabel("recovered N")
        plt.plot(x,x,'r--')

        plt.legend()
        plt.savefig('N.png')
        plt.savefig('N.pdf')
        #plt.scatter(true_Ns,N_ratios,label="N")


    def plot_accuracy_exp(self):
        """
        Plot the accuracy of the results
        """
        true_centres = []

        for fn,centre,errors,yaml_file in zip(self.filename,self.means,self.stds,self.yaml_files):
            #load the yaml file
            with open(yaml_file) as f:
                yaml_data = yaml.safe_load(f)
                true_centres.append(np.array([yaml_data['k'],yaml_data['N']]))
        #plot the first element of the ratios
        true_ks = [r[0] for r in true_centres]
        ks = [r[0] for r in self.means]
        k_errs = [r[0] for r in self.stds]

        true_Ns = [r[1] for r in true_centres]
        Ns = [r[1] for r in self.means]
        N_errs = [r[1] for r in self.stds]
        fitted_k = []
        fitted_k_err = []
        #estimate k for each set of bins
        for bin_centers,corrected_heights in zip(self.snr_bin_midpoints,self.simp_selection_corrected_heights):
            #fit an exponential to the data
            from scipy.optimize import curve_fit
            def exp_func(x,A,k):
                return A*np.exp(-k*x)
            popt,pcov = curve_fit(exp_func,bin_centers,corrected_heights,p0=[1,1],maxfev=1000000,bounds=([0,0],[np.inf,10]))
            fitted_k.append(popt[1])
            fitted_k_err.append(np.sqrt(np.diag(pcov))[1])
            #integrate under the fitted k line
            print(popt)
            print(np.sqrt(np.diag(pcov)))

        plt.figure()
        max_k = max([max(true_ks),max(ks)])
        min_k = min([min(true_ks),min(ks)])
        plt.errorbar(true_ks,ks,yerr=k_errs,label="LuNfit",linestyle='None',marker='o')
        plt.errorbar(true_ks,fitted_k,yerr=fitted_k_err,label="Simple Correction",linestyle='None',marker='x')
        plt.legend()
        x = np.linspace(min_k,max_k,100)
        plt.plot(x,x,'r--')
        plt.xlabel(r"True $k$")
        plt.ylabel(r"recovered $k$")
        plt.savefig('ks.png')
        plt.savefig('ks.pdf')
        plt.close()
        # plt.show()
        plt.figure()
        max_N = max([max(true_Ns),max(Ns)])
        x = np.linspace(0,max_N,100)
        plt.errorbar(true_Ns,Ns,yerr=N_errs,label="LuNfit",linestyle='None',marker='o')
        plt.scatter(true_Ns,self.simp_total_N,label="Simple Correction",linestyle='None',marker='x',color='r')
        plt.legend()
        plt.xlabel("True N")
        plt.ylabel("recovered N")
        plt.plot(x,x,'r--')
        plt.savefig('N.png')
        plt.savefig('N.pdf')
        #plt.scatter(true_Ns,N_ratios,label="N")
        plt.legend()
        plt.close()

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', help='filenames to plot')
    parser.add_argument('-p', help="period", type=float ,default = -1)
    parser.add_argument('--obs_time',help="observation time",type=float,default=1)
    parser.add_argument("--plot_accuracy",help="plot the accuracy of the results",action="store_true")
    parser.add_argument("--exp",help="making the plots for an exponential distribution",action="store_true")
    parser.add_argument("--det_curve",help="inj_stats dill file",required=True)
    args = parser.parse_args()
    det_curve = args.det_curve
    period = args.p
    obs_time = args.obs_time
    plot_accuracy = args.plot_accuracy
    filenames = args.filenames
    dp = dynesty_plot(filenames)

    snr_thresh = statistics_basic.load_detection_fn(det_curve,min_snr_cutoff=1.6)
    dp.load_filenames()
    dp.load_detections()

    if args.exp:
        dp.plot_corner(labels=[r"$K$","N"],plot=False)
        if plot_accuracy:
            dp.plot_accuracy_exp()
    else:
        dp.plot_corner(labels=[r"$\mu$",r"$\sigma$","N"],plot=False)
        if plot_accuracy:
            dp.plot_accuracy_logn()
