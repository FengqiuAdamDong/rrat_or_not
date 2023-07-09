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
            dyplot.cornerplot(d.results, labels=labels,show_titles=True)
            mean,cov = dynesty.utils.mean_and_cov(d.results.samples,d.results.importance_weights())
            diag_cov = np.diag(cov)
            std = np.sqrt(diag_cov)
            means.append(cov)
            stds.append(std)
        self.means = np.array(covs)
        self.stds = np.array(stds)

    def plot_accraucy(self):
        """
        Plot the accuracy of the results
        """
        ratios = []
        for fn,centre,errors in zip(self.filename,self.means,self.stds):
            #get the mu and sigma from the filename
            split = fn.split('_')
            sigma = float(split[-1])
            mu = float(split[-2])
            N = float(split[-3])
            true_centre = np.array([mu,sigma,N])
            ratios.append(centre/true_centre)
        #plot the first element of the ratios
        mu_ratios = [r[0] for r in ratios]
        true_mus = [r[0] for r in true_centres]
        sigma_ratios = [r[1] for r in ratios]
        true_sigmas = [r[1] for r in true_centres]
        N_ratios = [r[2] for r in ratios]
        true_Ns = [r[2] for r in true_centres]

        plt.scatter(true_mus,mu_ratios,legend="mu")
        plt.scatter(true_sigmas,sigma_ratios,legend="sigma")
        plt.scatter(true_Ns,N_ratios,legend="N")
        plt.show()
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
    import pdb; pdb.set_trace()
