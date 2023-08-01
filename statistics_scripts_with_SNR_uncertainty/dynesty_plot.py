#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import dynesty
import sys
from bayes_factor_NS_LN_no_a_single import loglikelihood
from bayes_factor_NS_LN_no_a_single import pt_Uniform_N
from scipy.interpolate import RegularGridInterpolator
import yaml
import smplotlib
def Ntonull(N):
    """
    Convert the number of events to the nulling fraction
    """
    nulling_frac = 1-(N/(obs_time/period))
    # print("nulling frac",nulling_frac,"N",N)
    return nulling_frac
def nulltoN(null):
    N = (1-null)*(obs_time/period)
    # print(obs_time,period,"obs time, period")
    # print("N",N,"nulling frac",null)
    return N

def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

class dynesty_plot:
    """
    Class to plot dynesty results"""

    def __init__(self, filenames):
        self.filename = filenames
        #get the parameters from the first filename
        split = filenames[0].split('_')
        self.dist = split[1]
        #make self.dist lower case
        self.dist = self.dist.lower()
        self.dets = int(split[2])

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

    def plot_bayes_ratio_logn(self):
        mu_arr= []
        evidence_diff_arr = []
        evidence_diff_err = []
        for fn,logn_sampler in zip(self.filename,self.data):
            #get the logn version of the filename
            split = fn.split('_')
            exp_fn = '_'.join(split[:-1])+'_exp.h5'
            exp_sampler = dynesty.NestedSampler.restore(exp_fn)
            #get the logn evidence
            exp_evidence = exp_sampler.results.logz[-1]
            exp_evidence_error = exp_sampler.results.logzerr[-1]

            logn_evidence = logn_sampler.results.logz[-1]
            logn_evidence_error = logn_sampler.results.logzerr[-1]
            #propagate the errors
            evidence_error = np.sqrt(exp_evidence_error**2+logn_evidence_error**2)

            print("exp evidence",exp_evidence,"logn evidence",logn_evidence,"comparison",logn_evidence-exp_evidence,"error",evidence_error)
            #get the k value from fn
            mu = float(fn.split('_')[3])
            mu_arr.append(mu)
            evidence_diff_arr.append(logn_evidence-exp_evidence)
        plt.figure()
        plt.errorbar(mu_arr,evidence_diff_arr,yerr=evidence_error,linestyle='none')
        plt.xlabel(r"$mu$")
        plt.ylabel(r"$\ln(Z_{logn})-\ln(Z_{exp})$")
        plt.savefig(f"self.dets{self.dets}_evidence_diff_logn.pdf")


    def plot_bayes_ratio_exp(self):
        k_arr= []
        evidence_diff_arr = []

        for fn,exp_sampler in zip(self.filename,self.data):
            #get the logn version of the filename
            split = fn.split('_')
            logn_fn = '_'.join(split[:-1])+'_logn.h5'
            logn_sampler = dynesty.NestedSampler.restore(logn_fn)
            #get the logn evidence
            exp_evidence = exp_sampler.results.logz[-1]
            exp_evidence_error = exp_sampler.results.logzerr[-1]

            logn_evidence = logn_sampler.results.logz[-1]
            logn_evidence_error = logn_sampler.results.logzerr[-1]

            evidence_error = np.sqrt(exp_evidence_error**2+logn_evidence_error**2)
            print("exp evidence",exp_evidence,"logn evidence",logn_evidence,"comparison",exp_evidence-logn_evidence,"error",evidence_error)
            #get the k value from fn
            k = float(fn.split('_')[3])
            k_arr.append(k)
            evidence_diff_arr.append(exp_evidence-logn_evidence)
        plt.figure()
        plt.errorbar(k_arr,evidence_diff_arr,yerr=evidence_error,linestyle='none')
        plt.xlabel(r"$k$")
        plt.ylabel(r"$\ln(Z_{exp})-\ln(Z_{logn})$")
        plt.savefig(f"self.dets{self.dets}_evidence_diff_exp.pdf")


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
                plt.savefig(f.strip(".h5")+"_corner.pdf")
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
            split = fn.split('_')
            #join all but the last element
            yaml_file = '_'.join(split[:-1])+'.yaml'
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
        plt.title(f"{self.dets} detections true sigma = {true_sigmas[0]}")
        plt.savefig(f"{self.dets}_mus.pdf")
        plt.figure()
        plt.errorbar(true_sigmas,sigmas,yerr=sigma_errs,label="sigma",linestyle='None',marker='o')
        max_sigma = max([max(true_sigmas),max(sigmas)])
        x = np.linspace(0,max_sigma,100)
        plt.plot(x,x,'r--')
        plt.xlabel(r"True $\sigma$")
        plt.ylabel(r"recovered $\sigma$")
        plt.title(f"{self.dets} detections true sigma = {true_sigmas[0]}")
        plt.savefig(f"{self.dets}_sigmas.pdf")
        plt.figure()
        max_N = max([max(true_Ns),max(Ns)])
        x = np.linspace(0,max_N,100)
        plt.errorbar(true_Ns,Ns,yerr=N_errs,label="N",linestyle='None',marker='o')
        plt.xlabel("True N")
        plt.ylabel("recovered N")
        plt.title(f"{self.dets} detections true sigma = {true_sigmas[0]}")
        plt.plot(x,x,'r--')
        plt.savefig(f"{self.dets}_logn_Ns.pdf")
        #plt.scatter(true_Ns,N_ratios,label="N")
        plt.legend()

    def plot_accuracy_exp(self):
        """
        Plot the accuracy of the results
        """
        true_centres = []

        for fn,centre,errors in zip(self.filename,self.means,self.stds):
            #get the mu and sigma from the filename
            split = fn.split('_')
            #join all but the last element
            yaml_file = '_'.join(split[:-1])+'.yaml'
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
        plt.xlabel(r"True $k$")
        plt.ylabel(r"recovered $k$")
        plt.title(f"{self.dets} detections")
        plt.savefig(f"{self.dets}_ks.pdf")

        plt.figure()
        max_N = max([max(true_Ns),max(Ns)])
        x = np.linspace(0,max_N,100)
        plt.errorbar(true_Ns,Ns,yerr=N_errs,label="N",linestyle='None',marker='o')
        plt.xlabel("True N")
        plt.ylabel("recovered N")
        plt.title(f"{self.dets} detections")
        plt.plot(x,x,'r--')
        plt.savefig(f"{self.dets}_exp_Ns.pdf")
        #plt.scatter(true_Ns,N_ratios,label="N")
        plt.legend()

    def plot_fit_logn(self):
        import statistics_basic
        for f,m in zip(self.filename,self.means):
            #get the mu and sigma from the filename
            split = f.split('_')
            #join all but the last element
            yaml_file = '_'.join(split[:-1])+'.yaml'
            real_det = '_'.join(split[:-1])+'.dill'
            #load the yaml file
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            detection_curve = data['detection_curve']
            snr_thresh = data['snr_thresh']
            if snr_thresh < 1.6:
                snr_thresh = 1.6
            #load the detection curve
            snr_thresh = statistics_basic.load_detection_fn(detection_curve,min_snr_cutoff=snr_thresh)
            import statistics
            from bayes_factor_NS_LN_no_a_single import process_detection_results
            #load the detections
            det_fluence, det_width, det_snr, noise_std = process_detection_results(real_det)
            det_snr = det_snr[det_snr > snr_thresh]
            fit_x = np.linspace(1e-9, 10, 10000)
            mu = m[0]
            std = m[1]
            det_error = statistics.det_error
            fit_y, p_det, conv_amp_array, conv = statistics.first_plot(fit_x, mu, std, det_error, a=0)
            fit_y = smoothTriangle(fit_y, 100)
            fit_y = fit_y / np.trapz(fit_y, fit_x)
            fig, ax = plt.subplots(1, 1)
            ax.hist(det_snr, bins='auto', density=True, label=f"max_mu={m[0]:.2f}, max_std={m[1]:.2f}")
            ax.plot(fit_x, fit_y, label="fit")
            # ax.plot(conv_amp_array, conv, label="convolution")
            # ax.plot(conv_amp_array, p_det, label="p_det")
            ax.set_xlabel("SNR")
            ax.set_ylabel("Probability")
            # plt.legend()
            plt.show()

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', help='filenames to plot')
    parser.add_argument('-p', help="period", type=float ,default = -1)
    parser.add_argument('--obs_time',help="observation time",type=float,default=1)
    parser.add_argument("--plot_accuracy",help="plot the accuracy of the results",action="store_true")
    parser.add_argument("--exp",help="making the plots for an exponential distribution",action="store_true")
    parser.add_argument("--bayes_ratio",help="plot the bayes ratio",action="store_true")

    args = parser.parse_args()
    global period
    global obs_time
    period = args.p
    obs_time = args.obs_time
    plot_accuracy = args.plot_accuracy
    filenames = args.filenames
    dp = dynesty_plot(filenames)
    dp.load_filenames()
    if args.exp:
        dp.plot_corner(labels=[r"$K$","N"],plot=True)
        if plot_accuracy:
            dp.plot_accuracy_exp()
        if args.bayes_ratio:
            dp.plot_bayes_ratio_exp()

    else:
        dp.plot_corner(labels=[r"$\mu$",r"$\sigma$","N"],plot=True)
        #dp.plot_fit_logn()
        if plot_accuracy:
            dp.plot_accuracy_logn()
        if args.bayes_ratio:
            dp.plot_bayes_ratio_logn()
