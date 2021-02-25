import numpy as np
import emcee
import matplotlib.pyplot as plt


def log_prob(x, mu, cov):
    diff = x - mu
    # dot product of diff, and solution to cov=diff
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))


ndim = 5

np.random.seed(42)
# generate 5 array of 5 random numbers
means = np.random.rand(ndim)

# generate 5x5 array of random numbers
# I'm not sure what the reshape is for
cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
# retrieve upper triangle
cov = np.triu(cov)
# transpose, subtract off the diagonal, and add to orig
cov += cov.T - np.diag(cov.diagonal())
# dot product it, now it should be a proper covariance matrix
cov = np.dot(cov, cov)

# choose to use 32 walkers
nwalkers = 32
# make initial guess, between 0-1 to start
p0 = np.random.rand(nwalkers, ndim)

# set up sampler
# function will be called such that
# log_prob(p0[0], means, cov)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[means, cov])

# burn in the MCMC, for 100 steps (arbitrary # of steps)
# save the final position of the burn in
state = sampler.run_mcmc(p0, 100)

# check what it looks like with just burning in
plt.figure('just burn-in')
samples = sampler.get_chain(flat=True)
plt.hist(samples[:, 0], 100, color='k', histtype='step')
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([])
# reset all of the bookkeeping parameters, as well as the walker positions
sampler.reset()

# for real, now (production run)
# start with the 'burned in' state
sampler.run_mcmc(state, 100)

# plot it!
plt.figure('with burnin, 100 runs')
samples = sampler.get_chain(flat=True)
plt.hist(samples[:, 0], 100, color='k', histtype='step')
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([])


sampler.run_mcmc(state, 10000)

# plot it!
plt.figure('10000 runs')
samples = sampler.get_chain(flat=True)
plt.hist(samples[:, 0], 100, color='k', histtype='step')
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([])


# check whether things went well
print("Mean acceptance fraction: {0:.3f}".format(
    np.mean(sampler.acceptance_fraction)))

print(
    "Mean autocorrelation time: {0:.3f} steps".format(
        np.mean(sampler.get_autocorr_time())
    )
)


