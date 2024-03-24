import numpy as np
import matplotlib.pyplot as plt
import dynesty
from bayes_factor_NS_LN_no_a_single import loglikelihood
from bayes_factor_NS_LN_no_a_single import pt_Uniform_N
from dynesty import plotting as dyplot
import smplotlib

class post_process:
    def __init__(self, fns):
        self.fns = fns
        for fn in fns:
            if 'expexp' in fn:
                self.expexpfn = fn
            elif 'lnln' in fn:
                self.lnlnfn = fn
            elif 'expln' in fn:
                self.explnfn = fn
            elif 'lnexp' in fn:
                self.lnexpfn = fn
        #load data
        self.expexp_results = self.load_data(self.expexpfn)
        self.lnln_results = self.load_data(self.lnlnfn)
        self.expln_results = self.load_data(self.explnfn)
        self.lnexp_results = self.load_data(self.lnexpfn)


    def load_data(self,fn):
        data = np.load(fn, allow_pickle=True)['results'].tolist()
        return data

    def plot_corner(self, ):
        label_expexp = ['k1','k2','N']
        fig, axes = dyplot.cornerplot(self.expexp_results, labels=label_expexp)
        plt.savefig('expexp_corner.png')
        plt.close()
        label_lnln = ['mu1','std1','mu2','std2','N']
        fig, axes = dyplot.cornerplot(self.lnln_results, labels=label_lnln)
        plt.savefig('lnln_corner.png')
        plt.close()
        label_expln = ['k1','mu2','std2','N']
        fig, axes = dyplot.cornerplot(self.expln_results, labels=label_expln)
        plt.savefig('expln_corner.png')
        plt.close()
        label_lnexp = ['mu1','std1','k2','N']
        fig, axes = dyplot.cornerplot(self.lnexp_results, labels=label_lnexp)
        plt.savefig('lnexp_corner.png')
        plt.close()

    def plot_bayes_ratio(self, ):
        evidence_expexp = self.expexp_results['logz'][-1]
        evidence_expexp_err = self.expexp_results['logzerr'][-1]
        evidence_lnln = self.lnln_results['logz'][-1]
        evidence_lnln_err = self.lnln_results['logzerr'][-1]
        evidence_expln = self.expln_results['logz'][-1]
        evidence_expln_err = self.expln_results['logzerr'][-1]
        evidence_lnexp = self.lnexp_results['logz'][-1]
        evidence_lnexp_err = self.lnexp_results['logzerr'][-1]

        plt.errorbar([0,1,2,3], [evidence_expexp, evidence_lnln, evidence_expln, evidence_lnexp], yerr=[evidence_expexp_err, evidence_lnln_err, evidence_expln_err, evidence_lnexp_err], fmt='o')
        plt.xticks([0,1,2,3], ['expexp', 'lnln', 'expln', 'lnexp'])
        plt.ylabel('log evidence')
        plt.savefig('log_evidence.png')
        plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', nargs='+' , help='filename')
    args = parser.parse_args()
    fns = args.fn

    pp = post_process(fns)
    pp.plot_corner()
    pp.plot_bayes_ratio()
