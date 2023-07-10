#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import dynesty
import sys
from bayes_factor_NS_LN_no_a_single import loglikelihood
from bayes_factor_NS_LN_no_a_single import pt_Uniform_N
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

    def plot_corner(self, labels=None):
        """
        Plot the corner plot of the samples
        """
        from dynesty import plotting as dyplot
        means = []
        stds = []
        for d in self.data:
            #dyplot.cornerplot(d.results, labels=labels,show_titles=True)
            mean,cov = dynesty.utils.mean_and_cov(d.results.samples,d.results.importance_weights())
            diag_cov = np.diag(cov)
            std = np.sqrt(diag_cov)
            means.append(mean)
            stds.append(std)
        self.means = np.array(means)
        self.stds = np.array(stds)

    def plot_accuracy(self):
        """
        Plot the accuracy of the results
        """
        ratios = []
        true_centres = []
        for fn,centre,errors in zip(self.filename,self.means,self.stds):
            #get the mu and sigma from the filename
            split = fn.split('_')
            sigma = float(split[-2])
            mu = float(split[-3])
            N = float(split[-4])
            true_centre = np.array([mu,sigma,N])
            true_centres.append(true_centre)
            ratios.append(centre/true_centre)
        #plot the first element of the ratios
        mu_ratios = [r[0] for r in ratios]
        true_mus = [r[0] for r in true_centres]
        mus = [r[0] for r in self.means]
        sigma_ratios = [r[1] for r in ratios]
        true_sigmas = [r[1] for r in true_centres]
        sigmas = [r[1] for r in self.means]
        N_ratios = [r[2] for r in ratios]
        true_Ns = [r[2] for r in true_centres]
        Ns = [r[2] for r in self.means]
        plt.figure()
        plt.scatter(true_mus,mus,label="mu")
        x = np.linspace(-1,2,100)
        plt.plot(x,x,'r--')
        plt.savefig('mus.png')
        plt.figure()
        plt.scatter(true_sigmas,sigmas,label="sigma")
        x = np.linspace(0,1,100)
        plt.plot(x,x,'r--')
        plt.savefig('sigmas.png')
        plt.figure()
        x = np.linspace(0,5e6)
        plt.scatter(true_Ns,Ns,label="N")
        #plt.plot(x,x,'r--')
        plt.savefig('N.png')

        #plt.scatter(true_Ns,N_ratios,label="N")
        plt.legend()
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', help='filenames to plot')

    args = parser.parse_args()
    filenames = args.filenames
    dp = dynesty_plot(filenames)
    dp.load_filenames()
    dp.plot_corner()
    dp.plot_accuracy()
