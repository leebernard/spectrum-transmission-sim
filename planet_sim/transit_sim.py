import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.optimize import minimize

from planet_sim.transit_toolbox import alpha_lambda
from planet_sim.transit_toolbox import open_cross_section
from planet_sim.transit_toolbox import gen_measured_transit
from planet_sim.transit_toolbox import transit_spectra_model
from toolkit import instrument_non_uniform_tophat
from toolkit import improved_non_uniform_tophat
from toolkit import consecutive_mean

'''
generate absorption profile from cross section data

possible issues with this simulation:
cross section changing with T is not accounted for
Gravity is assumed to be constant (thin shell approximation)
Everything is 1D...

Future expansions needed:
Account for temperature structure in scale height
'''

# define some housekeeping variables
wn_start = 4000
wn_end = 10000

# open these files carefully, because they are potentially over 1Gb in size
water_data_file = './line_lists/H2O_30mbar_1500K.txt'
water_wno, water_cross_sections = open_cross_section(water_data_file, wn_range=(wn_start, wn_end))

co_data_file = './line_lists/CO_30mbar_1500K'
co_wno, co_cross_sections = open_cross_section(co_data_file, wn_range=(wn_start, wn_end))

hcn_data_file = './line_lists/HCN_30mbar_1500K'
hcn_wno, hcn_cross_sections = open_cross_section(hcn_data_file, wn_range=(wn_start, wn_end))

h2_data_file = './line_lists/H2H2_CIA_30mbar_1500K.txt'
h2_wno, h2_cross_sections = open_cross_section(h2_data_file, wn_range=(wn_start, wn_end))

# interpolate the two different wavenumbers to the same wavenumber
fine_wave_numbers = np.arange(wn_start, wn_end, 2.0)
water_cross_sections = 10**np.interp(fine_wave_numbers, water_wno, np.log10(water_cross_sections))
co_cross_sections = 10**np.interp(fine_wave_numbers, co_wno, np.log10(co_cross_sections))
hcn_cross_sections = 10**np.interp(fine_wave_numbers, hcn_wno, np.log10(hcn_cross_sections))
h2_cross_sections = 10**np.interp(fine_wave_numbers, h2_wno, np.log10(h2_cross_sections))

# convert wavenumber to wavelength in microns
fine_wavelengths = 1e4/fine_wave_numbers


# plot them to check
plt.figure('compare_cross_section')
plt.plot(fine_wavelengths, water_cross_sections, label='H2O')
plt.plot(fine_wavelengths, co_cross_sections, label='CO')
plt.plot(fine_wavelengths, hcn_cross_sections, label='HCN')
plt.plot(fine_wavelengths, h2_cross_sections, label='H2')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Cross section (cm^2/molecule)')
plt.yscale('log')
plt.legend()

#
# # using data from Gliese 876 d, pulled from Wikipedia
# rad_planet = 1.65  # earth radii
# rad_star = .376  # solar radii
# m_planet = 6.8  # in earth masses
#
# # data from Kepler-10c
# rad_planet = 2.35
# m_planet = 7.37
#
# # fuck it, use made up shit
# # assuming relationship of r = m^0.55
# rad_planet = 3.5
# m_planet = 10
#
# # reference pressure: 1 barr
# p0 = 1  # bars
# T = 290  # K
# mass = 18  # amu
#

"""
# hot jupiter time!
# based upon KELT-11b, taken from Beatty et al 2017
rad_planet = 1.47  # in jovian radii
m_planet = 0.235  # jovian masses
rad_star = 2.94
p0 = 1
# temperature is made up
T = 1500
"""
# based upon HD209458b
rad_planet = 1.38  # in jovian radii
g_planet = 9.3  # m/s
rad_star = 1.161  # solar radii
p0 = 1  # barr
# temperature is made up
T = 1500



# water ratio taken from Chageat et al 2020
# water_ratio = 2600. * 1e-6  # in parts per million

# use log ratio instead
# log_f_h2o = np.log10(water_ratio)
log_f_h2o = -3  # 1000ppm
log_fco = -3
log_fhcn = -5  # 10ppm

# resolution of spectrograph
R = 30

# generate wavelength sampling of spectrum
# flip the data to ascending order
flipped_wl = np.flip(fine_wavelengths)
resolution = np.mean(flipped_wl)/R
# Choose the Nyquest sampling rate at the blue end
samplerate_per_pixel = resolution/2
number_pixels = int((flipped_wl[-1] - flipped_wl[0]) / samplerate_per_pixel)
# make pixel bins
pixel_bins = np.linspace(flipped_wl[0], flipped_wl[-1], num=number_pixels+1)


# pixel_wavelengths = np.linspace(flipped_wl[0], flipped_wl[-1], num=number_pixels)

# pixel_bins = pixel_wavelengths
# test the model generation function
# this produces the 'true' transit spectrum
fixed_parameters = (fine_wavelengths,
                    water_cross_sections,
                    co_cross_sections,
                    hcn_cross_sections,
                    h2_cross_sections,
                    g_planet,
                    rad_star,
                    R)
variables = (rad_planet,
             T,
             log_f_h2o,
             log_fco,
             log_fhcn)
pixel_wavelengths, pixel_transit_depth = transit_spectra_model(pixel_bins, variables, fixed_parameters)

# generate photon noise from a signal value
# signal = 1.22e9
# noise = (np.random.poisson(lam=signal, size=pixel_transit_depth.size) - signal)/signal

photon_noise = 75 * 1e-6  # set noise to 75ppm
noise = np.random.normal(scale=photon_noise, size=pixel_transit_depth.size)

# add noise to the transit spectrum
noisey_transit_depth = pixel_transit_depth + noise

# mean spectral resolution
# spec_res = resolution

plt.figure('transit depth R%.2f' %R, figsize=(8, 8))
plt.subplot(212)
plt.plot(fine_wavelengths, np.flip(water_cross_sections), label='H2O')
plt.plot(fine_wavelengths, np.flip(co_cross_sections), label='CO')
plt.plot(fine_wavelengths, np.flip(hcn_cross_sections), label='HCN')
plt.plot(fine_wavelengths, np.flip(h2_cross_sections), label='H2')
plt.title('Absorption Cross section')
plt.legend()
plt.xlabel('Wavelength (μm)')
plt.ylabel('Cross section (cm^2/molecule)')
plt.yscale('log')

plt.subplot(211)
plt.plot(pixel_wavelengths, pixel_transit_depth, label='Ideal')
plt.errorbar(pixel_wavelengths, noisey_transit_depth, yerr=photon_noise, label='Photon noise', fmt='o', capsize=2.0)
plt.title('Transit depth, R= %d, water= %d ppm' % (R, 10**log_f_h2o/1e-6) )
plt.legend(('Ideal', 'Photon noise'))
plt.ylabel('($R_p$/$R_{star}$)$^2$ (%)')
plt.subplot(211).yaxis.set_major_formatter(FormatStrFormatter('% 1.1e'))



'''Fit the data'''


# define a likelyhood function
def log_likelihood(theta, x, y, yerr, fixed):

    _, model = transit_spectra_model(x, theta, fixed)

    # print('model.size', model.size)
    # print('y.size', y.size)
    sigma = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma + np.log(sigma))


# define a prior function
def log_prior(theta):
    '''Basically just saying, the fixed_parameters are within these values'''
    rad_planet, T, log_f_h2o, log_fco, log_fhcn = theta
    if 0.0 < rad_planet < 10 and 300 < T < 3000.0 and \
            -12 < log_f_h2o < -1.0 and -12 < log_fco < -1.0 and -12 < log_fhcn < -1.0:
        return 0.0
    else:
        return -np.inf

def prior_transform(u):
    # u is random samples from the unit cube
    x = np.array(u)  # copy u

    # planet radius prior
    x[0] = u[0]*10
    # Temperature
    x[1] = u[1]*(3000-300) + 300

    # set the trace species to uniform priors
    x[2:-1] = u[2:-1]*11 - 12

    return x


def log_probability(theta, x, y, yerr, fixed):
    '''full probability function
    If fixed_parameters are withing the range defined by log_prior, return the likelihood.
    otherwise, return a flag'''
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        # why are they summed?
        # oh, because this is log space. They would be multiplied if this was base space
        return lp + log_likelihood(theta, x, y, yerr, fixed)



yerr = photon_noise

np.random.seed(42)
nll = lambda *args: -log_probability(*args)
# create initial guess from true values, by adding a little noise
initial = np.array(variables) + variables*(0.1*np.random.randn(5))
# find the fixed_parameters with maximized likelyhood, according to the given distribution function
soln = minimize(nll, initial, args=(pixel_bins, pixel_transit_depth, yerr, fixed_parameters))
# unpack the solution
rad_ml, T_ml, waterfrac_ml, cofrac_ml, hcn_frac_ml = soln.x

print("Maximum likelihood estimates: (true)")
print("Rad_planet = {0:.3f} ({1:.3f})".format(rad_ml, rad_planet))
print("T = {0:.3f} ({1:.3f})".format(T_ml, T))
print("log_water_ratio = {0:.3f} ({1:.3f})".format(waterfrac_ml, log_f_h2o))
print("log_co_ratio = {0:.3f} ({1:.3f})".format(cofrac_ml, log_fco))
print("log_hcn_ratio = {0:.3f} ({1:.3f})".format(hcn_frac_ml, log_fhcn))

theta_fit = soln.x
fit_wavelengths, fit_transit = transit_spectra_model(pixel_bins, theta_fit, fixed_parameters)

plt.figure('fit_result')
plt.subplot(111)
plt.plot(pixel_wavelengths, pixel_transit_depth, label='Ideal')
plt.errorbar(pixel_wavelengths, noisey_transit_depth, yerr=photon_noise, label='Photon noise', fmt='o', capsize=2.0)
plt.plot(fit_wavelengths, fit_transit, label='Fit result')
plt.title('Transit depth, R= %d, water= %d ppm' % (R, 10**log_f_h2o/1e-6))
plt.legend()
plt.ylabel('($R_p$/$R_{star}$)$^2$ (%)')
plt.subplot(111).yaxis.set_major_formatter(FormatStrFormatter('% 1.1e'))


from multiprocessing import Pool


# generate 32 walkers, with small gaussian deviations from minimization soln
pos = soln.x + soln.x*(1e-3 * np.random.randn(32, 5))
nwalkers, ndim = pos.shape

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    log_probability,
                                    args=(pixel_bins, pixel_transit_depth, yerr, fixed_parameters),
                                    pool=pool)
    sampler.run_mcmc(pos, 10000, progress=True)

# examine the results
fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["Rad_planet", "T", "water_ratio", "co_ratio", "hcn_ratio"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")


# hard to say how quickly it filled the posterior distribution from the tiny prior walkers
# but we can look at estimate of integrated autocorrelation time (whatever that means)
tau = sampler.get_autocorr_time()
print(tau)


# examine it with the initial burn-in period discarded
# also, what does thinning the autocorrelation time do?
flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
print(flat_samples.shape)


# generate a corner plot
# import corner

fig = corner.corner(flat_samples, labels=labels, truths=[rad_planet, T, log_f_h2o, log_fco, log_fhcn])
fig.suptitle('Blue lines are true values', fontsize=14)
# fig.savefig('/test/my_first_cornerplot.png')




