

# # Python 3 compatability
# from __future__ import division, print_function
# from six.moves import range

# system functions that are always useful to have
import time, sys, os

# basic numeric setup
import numpy as np
import dynesty
# plotting
import matplotlib
from matplotlib import pyplot as plt

# seed the random number generator
np.random.seed(56101)

# # re-defining plotting defaults
# from matplotlib import rcParams
# rcParams.update({'xtick.major.pad': '7.0'})
# rcParams.update({'xtick.major.size': '7.5'})
# rcParams.update({'xtick.major.width': '1.5'})
# rcParams.update({'xtick.minor.pad': '7.0'})
# rcParams.update({'xtick.minor.size': '3.5'})
# rcParams.update({'xtick.minor.width': '1.0'})
# rcParams.update({'ytick.major.pad': '7.0'})
# rcParams.update({'ytick.major.size': '7.5'})
# rcParams.update({'ytick.major.width': '1.5'})
# rcParams.update({'ytick.minor.pad': '7.0'})
# rcParams.update({'ytick.minor.size': '3.5'})
# rcParams.update({'ytick.minor.width': '1.0'})
# rcParams.update({'font.size': 30})
#

# truth
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# generate mock data
N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y_true = m_true * x + b_true
y = y_true + np.abs(f_true * y_true) * np.random.randn(N)
y += yerr * np.random.randn(N)

# plot results
plt.figure(figsize=(10, 5))
plt.errorbar(x, y, yerr=yerr, fmt='ko', ecolor='red')
plt.plot(x, y_true, color='blue', lw=3)
plt.xlabel(r'$X$')
plt.ylabel(r'$Y$')
plt.tight_layout()


# log-likelihood
def loglike(theta):
    '''
    As far as I can tell, x, y, and yerr are defined outside the scope of this function
    '''
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))

    return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


# prior transform
def prior_transform(utheta):
    um, ub, ulf = utheta
    m = 5.5 * um - 5.
    b = 10. * ub
    lnf = 11. * ulf - 10.

    return m, b, lnf


dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=3,
                                        bound='multi', sample='rstagger')
dsampler.run_nested()
dres = dsampler.results


