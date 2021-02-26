"""
Learn to fit a line to data when I don't trust error bars
"""

import numpy as np
import matplotlib.pyplot as plt

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


from scipy.optimize import minimize
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





