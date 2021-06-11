import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.optimize import minimize

from planet_sim.transit_toolbox import alpha_lambda
from planet_sim.transit_toolbox import open_cross_section
from planet_sim.transit_toolbox import gen_measured_transit
from planet_sim.transit_toolbox import transit_spectra_model
from toolkit import instrument_non_uniform_tophat


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

h2_data_file = './line_lists/H2H2_CIA_30mbar_1500K.txt'
h2_wno, h2_cross_sections = open_cross_section(h2_data_file, wn_range=(wn_start, wn_end))

# interpolate the two different wavenumbers to the same wavenumber
fine_wave_numbers = np.arange(wn_start, wn_end, .1)
water_cross_sections = 10**np.interp(fine_wave_numbers, water_wno, np.log10(water_cross_sections))
h2_cross_sections = 10**np.interp(fine_wave_numbers, h2_wno, np.log10(h2_cross_sections))

# convert wavenumber to wavelength in microns
fine_wavelengths = 1e4/fine_wave_numbers


# plot them to check
plt.figure('compare_cross_section')
plt.plot(fine_wavelengths, water_cross_sections)
plt.plot(fine_wavelengths, h2_cross_sections)
plt.xlabel('Wavelength (μm)')
plt.ylabel('Cross section (cm^2/molecule)')
plt.yscale('log')
plt.legend(('H2O', 'H2'))

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

# hot jupiter time!
# based upon KELT-11b, taken from Beatty et al 2017
rad_planet = 1.47  # in jovian radii
m_planet = 0.235  # jovian masses
rad_star = 2.94
p0 = 1
# temperature is made up
T = 1500
# water ratio taken from Chageat et al 2020
water_ratio = 2600. * 1e-6  # in parts per million
# resolution of spectrograph
R = 70

# generate wavelength sampling of spectrum
# flip the data to ascending order
fine_wl = np.flip(fine_wavelengths)
resolution = np.mean(fine_wl)/R
# Choose the Nyquest sampling rate at the blue end
samplerate_per_pixel = resolution/2
number_pixels = int((fine_wl[-1] - fine_wl[0]) / samplerate_per_pixel)
pixel_wavelengths = np.linspace(fine_wl[0], fine_wl[-1], num=number_pixels)

# test the model generation function
fixed_parameters = fine_wavelengths, water_cross_sections, h2_cross_sections, m_planet, rad_star, R
variables = rad_planet, T, water_ratio
pixel_wavelengths, pixel_transit_depth = transit_spectra_model(pixel_wavelengths, variables, fixed_parameters)

# generate photon noise from a signal value
signal = 1.22e9
photon_noise = 1/np.sqrt(signal)  # calculate noise as fraction of signal
# noise = np.random.normal(scale=photon_noise, size=pixel_transit_depth.size)
noise = (np.random.poisson(lam=signal, size=pixel_transit_depth.size) - signal)/signal
# add noise to the transit depth
noisey_transit_depth = pixel_transit_depth + noise

# mean spectral resolution
# spec_res = resolution

plt.figure('transit depth R%.2f' %R, figsize=(8, 8))
plt.subplot(212)
plt.plot(fine_wavelengths, np.flip(water_cross_sections))
plt.plot(fine_wavelengths, np.flip(h2_cross_sections))
plt.title('Cross section of H2O')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Cross section (cm^2/molecule)')
plt.yscale('log')

plt.subplot(211)
plt.plot(pixel_wavelengths, pixel_transit_depth)
plt.errorbar(pixel_wavelengths, noisey_transit_depth, yerr=photon_noise, fmt='o', capsize=2.0)
plt.title('Transit depth, R= %d, water= %d ppm' % (R, water_ratio/1e-6) )
plt.legend(('Ideal', 'Photon noise'))
plt.ylabel('($R_p$/$R_{star}$)$^2$ (%)')
plt.subplot(211).yaxis.set_major_formatter(FormatStrFormatter('% 1.1e'))

'project_data/1H2-16O_6250-12500_300K_20.000000.sigma'
'project_data/1H2-16O_6250-12500_300K_100.000000.sigma'


'''Fit the data'''


# define a likelyhood function
def log_likelihood(theta, x, y, yerr, fixed):

    model = transit_spectra_model(x, theta, fixed)
    sigma = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma + np.log(sigma))


# define a prior function
def log_prior(theta):
    '''Basically just saying, the fixed_parameters are within these values'''
    rad_planet, T, water_ratio = theta
    if 0.0 < rad_planet < 10 and 0.0 < T < 5000.0 and 0 < water_ratio < 1.0:
        return 0.0
    else:
        return -np.inf


def log_probability(theta, x, y, yerr):
    '''full probability function
    If fixed_parameters are withing the range defined by log_prior, return the likelihood.
    otherwise, return a flag'''
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        # why are they summed?
        # oh, because this is log space. They would be multiplied if this was base space
        return lp + log_likelihood(theta, x, y, yerr)



yerr = photon_noise

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
# create initial guess from true values, by adding a little noise
initial = np.array(variables) + 0.1*np.random.randn(3)
# find the fixed_parameters with maximized likelyhood, according to the given distribution function
soln = minimize(nll, initial, args=(pixel_wavelengths, pixel_transit_depth, yerr, fixed_parameters))
# unpack the solution
rad_ml, T_ml, waterfrac_ml = soln.x

print("Maximum likelihood estimates:")
print("Rad_planet = {0:.3f}".format(rad_ml))
print("T = {0:.3f}".format(T_ml))
print("water_ratio = {0:.3f}".format(np.exp(waterfrac_ml)))

theta_fit = rad_ml, T_ml, waterfrac_ml
fit_transit = transit_spectra_model(pixel_wavelengths, variables, fixed_parameters)

plt.figure('fit_result')
plt.plot(pixel_wavelengths, pixel_transit_depth)
plt.errorbar(pixel_wavelengths, noisey_transit_depth, yerr=photon_noise, fmt='o', capsize=2.0)
plt.plot(pixel_wavelengths, pixel_transit_depth)
plt.title('Transit depth, R= %d, water= %d ppm' % (R, water_ratio/1e-6) )
plt.legend(('Ideal', 'Photon noise', 'Fit result'))
plt.ylabel('($R_p$/$R_{star}$)$^2$ (%)')
plt.subplot(211).yaxis.set_major_formatter(FormatStrFormatter('% 1.1e'))

