"""
Learn to fit a line to data when I don't trust error bars
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

from scipy.optimize import minimize


np.random.seed(123)

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model
# number of data points
N = 50
# produce random data on 0-10, ascending order
x = np.sort(10 * np.random.rand(N))
# generate set of error estimates for each data point
yerr = 0.1 + 0.5 * np.random.rand(N)
# generate y data from x data
y = m_true*x + b_true
# add unaccounted-for noise
y += np.abs(f_true * y) * np.random.randn(N)
# add accounted-for noise
y += yerr * np.random.randn(N)

# plot the data
plt.figure('fake data')
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
x0 = np.linspace(0, 10, 500)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")

'''now that the data is generated, play with it'''

# fit the data using linear least squares
A = np.vander(x, 2)
C = np.diag(yerr * yerr)
ATA = np.dot(A.T, A / (yerr ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))
print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))

# plot it
plt.figure('linear least sqs')
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")


'''maximum likelyhood estimation'''


# define a likelyhood function, which includes both measured error 'yerr'
# and amount the error is underestimated by, 'f'
def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m*x + b
    sigma2 = yerr**2 + model**2 * np.exp(2*log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


# from scipy.optimize import minimize
np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
# create initial guess from true values, by adding a little noise
initial = np.array([m_true, b_true, np.log(f_true)]) + 0.1*np.random.randn(3)
# find the parameters with maximized likelyhood, according to the given distribution function
soln = minimize(nll, initial, args=(x, y, yerr))
# unpack the solution
m_ml, b_ml, log_f_ml = soln.x

print("Maximum likelihood estimates:")
print("m = {0:.3f}".format(m_ml))
print("b = {0:.3f}".format(b_ml))
print("f = {0:.3f}".format(np.exp(log_f_ml)))

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
plt.plot(x0, np.dot(np.vander(x0, 2), [m_ml, b_ml]), ":k", label="ML")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")


'''
Now I want uncertainties on m and b, but don't really care much about f
MCMC will let me do both in one go
But first, I need to construct a prior function
'''


def log_prior(theta):
    '''Bascially just saying, the parameters are within these values'''
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10 < log_f < 1.0:
        return 0.0
    else:
        return -np.inf


def log_probability(theta, x, y, yerr):
    '''full probability function
    If parameters are withing the range defined by log_prior, return the likelihood.
    otherwise, return a flag'''
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        # why are they summed?
        return lp + log_likelihood(theta, x, y, yerr)


# import emcee

# generate 32 walkers, with small gaussian deviations from minimization soln
pos = soln.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, 5000, progress=True)


# examine the results
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b", "log(f)"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

# hard to say how quickly it filled the posterior distribution from the tiny prior walkers
# but we can look at estimate of ingerated autocorrelation time (whatever that means)
tau = sampler.get_autocorr_time()
print(tau)
# in this case, looks like about 40 steps needed to 'forget' where it started
# this is the 'burn in' time

# examine it with the initial burn-in period discarded
# also, what does thinning the autocorrelation time do?
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)


# generate a corner plot
# import corner

fig = corner.corner(flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)])
fig.suptitle('Corner plot of a meaningless line fit', fontsize=14)
fig.savefig('/test/my_first_cornerplot.png')


