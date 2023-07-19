#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import dynesty
import sys
from bayes_factor_NS_LN_no_a_single import loglikelihood
from bayes_factor_NS_LN_no_a_single import pt_Uniform_N
import yaml
import smplotlib
import statistics_basics

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

    def __init__(self, filenames)
        self.filename = filenames
        #strip the .h5 from the filename and all .dill
        self.detections = [f.strip(".h5")+".dill" for f in self.filename]

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

    def load_detections(self, num_bins=20):
        self.snrs = []
        self.snr_bin_midpoints = []
        self.snr_bin_heights = []
        self.simp_total_N = []
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
            simp_selection_corrected_heights = bin_heights/statistics_basics.p_detect_cpu(bin_midpoints)
            self.simp_total_N.append(np.sum(simp_selection_corrected_heights))


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

        for fn,centre,errors in zip(self.filename,self.means,self.stds):
            #get the mu and sigma from the filename
            split = fn.split('.')
            #join all but the last element
            yaml_file = '.'.join(split[:-1])+'.yaml'
            #load the yaml file
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

        plt.figure()
        max_mu = max([max(true_mus),max(mus)])
        min_mu = min([min(true_mus),min(mus)])
        plt.errorbar(true_mus,mus,yerr=mu_errs,label="mu",linestyle='None',marker='o')
        x = np.linspace(min_mu,max_mu,100)
        plt.plot(x,x,'r--')
        plt.xlabel(r"True $\mu$")
        plt.ylabel(r"recovered $\mu$")
        plt.savefig('mus.png')
        plt.figure()
        plt.errorbar(true_sigmas,sigmas,yerr=sigma_errs,label="sigma",linestyle='None',marker='o')
        max_sigma = max([max(true_sigmas),max(sigmas)])
        x = np.linspace(0,max_sigma,100)
        plt.plot(x,x,'r--')
        plt.xlabel(r"True $\sigma$")
        plt.ylabel(r"recovered $\sigma$")
        plt.savefig('sigmas.png')
        plt.figure()
        max_N = max([max(true_Ns),max(Ns)])
        x = np.linspace(0,max_N,100)
        plt.errorbar(true_Ns,Ns,yerr=N_errs,label="N",linestyle='None',marker='o')
        plt.xlabel("True N")
        plt.ylabel("recovered N")
        plt.plot(x,x,'r--')
        plt.savefig('N.png')
        #plt.scatter(true_Ns,N_ratios,label="N")
        plt.legend()

    def plot_accuracy_exp(self):
        """
        Plot the accuracy of the results
        """
        true_centres = []

        for fn,centre,errors in zip(self.filename,self.means,self.stds):
            #get the mu and sigma from the filename
            split = fn.split('.')
            #join all but the last element
            yaml_file = '.'.join(split[:-1])+'.yaml'
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

        plt.figure()
        max_k = max([max(true_ks),max(ks)])
        min_k = min([min(true_ks),min(ks)])
        plt.errorbar(true_ks,ks,yerr=k_errs,label="k",linestyle='None',marker='o')
        x = np.linspace(min_k,max_k,100)
        plt.plot(x,x,'r--')
        plt.xlabel(r"True $\k$")
        plt.ylabel(r"recovered $k$")
        plt.savefig('ks.png')

        plt.figure()
        max_N = max([max(true_Ns),max(Ns)])
        x = np.linspace(0,max_N,100)
        plt.errorbar(true_Ns,Ns,yerr=N_errs,label="N",linestyle='None',marker='o')
        plt.scatter(true_Ns,self.simp_totals,marker='x',label="simple correction")
        plt.xlabel("True N")
        plt.ylabel("recovered N")
        plt.plot(x,x,'r--')
        plt.savefig('N.png')
        #plt.scatter(true_Ns,N_ratios,label="N")
        plt.legend()


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
    dp.load_filenames()
    snr_thresh = statistics_basic.load_detection_fn(detection_curve,min_snr_cutoff=1.6)

    if args.exp:
        dp.plot_corner(labels=[r"$K$","N"],plot=False)
        if plot_accuracy:
            dp.plot_accuracy_exp()
    else:
        dp.plot_corner(labels=[r"$\mu$",r"$\sigma$","N"],plot=False)
        if plot_accuracy:
            dp.plot_accuracy_logn()
