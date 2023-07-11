#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import dynesty
import sys
from bayes_factor_NS_LN_no_a_single import loglikelihood
from bayes_factor_NS_LN_no_a_single import pt_Uniform_N
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

class dynesty_plot:
    """
    Class to plot dynesty results"""

    def __init__(self, filenames):
        self.filename = filenames

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
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', help='filenames to plot')
    parser.add_argument('--labels', nargs='+', help='labels for the plot')
    parser.add_argument('-p', help="period", type=float ,default = -1)
    parser.add_argument('--obs_time',help="observation time",type=float,default=1)
    parser.add_argument("--plot_accuracy",help="plot the accuracy of the results",action="store_true")
    args = parser.parse_args()
    global period
    global obs_time
    period = args.p
    obs_time = args.obs_time
    plot_accuracy = args.plot_accuracy
    filenames = args.filenames
    dp = dynesty_plot(filenames)
    dp.load_filenames()
    dp.plot_corner(labels=[r"$\mu$",r"$\sigma$","N"],plot=True)
    if plot_accuracy:
        dp.plot_accuracy_logn()
