import numpy as np
import matplotlib.pyplot as plt
import dynesty

from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import minimize
from dynesty import plotting as dyplot

from planet_sim.transit_toolbox import open_cross_section
from planet_sim.transit_toolbox import transit_spectra_model


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
wn_start = 5880  # 1.70068 um
# wn_end = 10000  # this is 1 um
wn_end = 9302  # 1.075 um

# open these files carefully, because they are potentially over 1Gb in size
water_data_file = './line_lists/H2O_30mbar_1500K.txt'
water_wno, water_cross_sections_raw = open_cross_section(water_data_file, wn_range=(wn_start, wn_end))

ch4_data_file = './line_lists/CH4_30mbar_1500K'
ch4_wno, ch4_cross_sections_raw = open_cross_section(ch4_data_file, wn_range=(wn_start, wn_end))

# co_data_file = './line_lists/CO_30mbar_1500K'
# co_wno, co_cross_sections = open_cross_section(co_data_file, wn_range=(wn_start, wn_end))

nh3_data_file = './line_lists/NH3_30mbar_1500K'
nh3_wno, nh3_cross_sections_raw = open_cross_section(nh3_data_file, wn_range=(wn_start, wn_end))

hcn_data_file = './line_lists/HCN_30mbar_1500K'
hcn_wno, hcn_cross_sections_raw = open_cross_section(hcn_data_file, wn_range=(wn_start, wn_end))

h2_data_file = './line_lists/H2H2_CIA_30mbar_1500K.txt'
h2_wno, h2_cross_sections_raw = open_cross_section(h2_data_file, wn_range=(wn_start, wn_end))

# interpolate the two different wavenumbers to the same wavenumber
fine_wave_numbers = np.arange(wn_start, wn_end, 3.0)

na_cross_sections =
k_cross_sections =
water_cross_sections = 10**np.interp(fine_wave_numbers, water_wno, np.log10(water_cross_sections_raw))
# co_cross_sections = 10**np.interp(fine_wave_numbers, co_wno, np.log10(co_cross_sections))
ch4_cross_sections = 10**np.interp(fine_wave_numbers, ch4_wno, np.log10(ch4_cross_sections_raw))
nh3_cross_sections = 10**np.interp(fine_wave_numbers, nh3_wno, np.log10(nh3_cross_sections_raw))
hcn_cross_sections = 10**np.interp(fine_wave_numbers, hcn_wno, np.log10(hcn_cross_sections_raw))
h2_cross_sections = 10**np.interp(fine_wave_numbers, h2_wno, np.log10(h2_cross_sections_raw))

# convert wavenumber to wavelength in microns
fine_wavelengths = 1e4/fine_wave_numbers


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
T = 1071

# use log ratio instead
# log_f_h2o = np.log10(water_ratio)
# taken from MacDonald 2017
log_fna = -5.13
log_fk =
log_f_h2o
log_fch4
log_fnh3
log_fhcn
# flip the data to ascending order
flipped_wl = np.flip(fine_wavelengths)
# see Hubble WFC3 slitless spectrograph in NIR
# https://hst-docs.stsci.edu/wfc3ihb/chapter-8-slitless-spectroscopy-with-wfc3/8-1-grism-overview

# this data based upon Deming et al 2013
# resolution of spectrograph
R=70

# open wavelength sampling of spectrum
sampling_data = './planet_sim/data/HD209458b_demingetal_data'
sampling_wl, err = open_cross_section(sampling_data)

pixel_delta_wl = np.diff(sampling_wl).mean()
# make pixel bins
# these bins are just a simple mean upsampling
# this is close enough for the purposes of this simulation
wfc3_start = sampling_wl[0] - pixel_delta_wl/2
wfc3_end = sampling_wl[-1] + pixel_delta_wl/2
pixel_bins = np.linspace(wfc3_start, wfc3_end, sampling_wl.size + 1)


# pixel_wavelengths = np.linspace(flipped_wl[0], flipped_wl[-1], num=number_pixels)

# pixel_bins = pixel_wavelengths
# test the model generation function
# this produces the 'true' transit spectrum
fixed_parameters = (fine_wavelengths,
                    na_cross_sections,
                    k_cross_sections,
                    water_cross_sections,
                    ch4_cross_sections,
                    nh3_cross_sections,
                    hcn_cross_sections,
                    h2_cross_sections,
                    g_planet,
                    rad_star,
                    R)
theta = (rad_planet,
         T,
         log_fna,
         log_fk,
         log_f_h2o,
         log_fch4,
         log_fnh3,
         log_fhcn)
pixel_wavelengths, pixel_transit_depth = transit_spectra_model(pixel_bins, theta, fixed_parameters)

# generate photon noise from a signal value
# signal = 1.22e9
# noise = (np.random.poisson(lam=signal, size=pixel_transit_depth.size) - signal)/signal

# generate noise instances
num_noise_inst = 10
photon_noise = 75 * 1e-6  # set noise to 75ppm
noise_inst = []
while len(noise_inst) < num_noise_inst:
 noise_inst.append( np.random.normal(scale=photon_noise, size=pixel_transit_depth.size) )

# add noise to the transit spectrum
noisey_transit_depth = pixel_transit_depth + noise_inst

plt.figure('transit depth R%.2f' %R, figsize=(8, 8))
plt.subplot(212)
plt.plot(flipped_wl, np.flip(water_cross_sections), label='H2O')
plt.plot(flipped_wl, np.flip(co_cross_sections), label='CO')
plt.plot(flipped_wl, np.flip(hcn_cross_sections), label='HCN')
plt.plot(flipped_wl, np.flip(h2_cross_sections), label='H2')
plt.title('Absorption Cross section')
plt.legend()
plt.xlabel('Wavelength (μm)')
plt.ylabel('Cross section (cm^2/molecule)')
plt.yscale('log')

plt.subplot(211)
plt.plot(pixel_wavelengths, pixel_transit_depth, label='Ideal')
plt.errorbar(pixel_wavelengths, noisey_transit_depth[0], yerr=photon_noise, label='Photon noise', fmt='o', capsize=2.0)
plt.title('Transit depth, R= %d, water= %d ppm' % (R, 10**log_f_h2o/1e-6) )
plt.legend(('Ideal', 'Photon noise'))
plt.ylabel('($R_p$/$R_{star}$)$^2$ (%)')
plt.subplot(211).yaxis.set_major_formatter(FormatStrFormatter('% 1.1e'))



'''Fit the data'''


# define a likelyhood function
def log_likelihood(theta):
    # retrieve the global variables
    global pixel_bins
    global transit_data
    global photon_noise
    global fixed_parameters
    # only 'y' changes on the fly
    fixed = fixed_parameters
    x = pixel_bins
    y = transit_data
    yerr = photon_noise

    _, model = transit_spectra_model(x, theta, fixed)

    sigma = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma + np.log(sigma))


# define a prior function
def prior_trans(u):
    # u is random samples from the unit cube
    x = np.array(u)  # copy u

    # planet radius prior
    x[0] = u[0]*10
    # Temperature
    x[1] = u[1]*(3000-300) + 300

    # set the trace species to uniform priors
    x[2:] = u[2:]*11 - 12

    # global print_number
    # if print_number < 100:
    #     print_number += 1
    #     print('parameter values:', x)
    return x


from multiprocessing import Pool

ndim = len(theta)
full_results = []
for transit_data in noisey_transit_depth:
    with Pool() as pool:
        sampler = dynesty.NestedSampler(log_likelihood, prior_trans, ndim,
                                        nlive=500, pool=pool, queue_size=pool._processes)
        sampler.run_nested()
        full_results.append(sampler.results)


from planet_sim.transit_toolbox import transit_spectra_h2o_only

# define a new prior, that uses only h20
# define a likelyhood function
def loglike_h2o_only(theta):
    # retrieve the global variables
    global pixel_bins
    global transit_data
    global photon_noise
    global fixed_parameters
    # only 'y' changes on the fly
    fixed = fixed_parameters
    x = pixel_bins
    y = transit_data
    yerr = photon_noise

    _, model = transit_spectra_h2o_only(x, theta, fixed)

    sigma = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma + np.log(sigma))


ndim = 3
h2o_results = []
for transit_data in noisey_transit_depth:
    with Pool() as pool:
        sampler = dynesty.NestedSampler(loglike_h2o_only, prior_trans, ndim,
                                        nlive=500, pool=pool, queue_size=pool._processes)
        sampler.run_nested()
        h2o_results.append(sampler.results)


from planet_sim.transit_toolbox import transit_spectra_no_h2o
# define a new prior, that uses only h20
# define a likelyhood function
def loglike_no_h2o(theta):
    # retrieve the global variables
    global pixel_bins
    global transit_data
    global photon_noise
    global fixed_parameters
    # only 'y' changes on the fly
    fixed = fixed_parameters
    x = pixel_bins
    y = transit_data
    yerr = photon_noise

    _, model = transit_spectra_no_h2o(x, theta, fixed)

    sigma = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma + np.log(sigma))

ndim = 4
co_hcn_results = []
for transit_data in noisey_transit_depth:
    with Pool() as pool:
        sampler = dynesty.NestedSampler(loglike_no_h2o, prior_trans, ndim,
                                        nlive=500, pool=pool, queue_size=pool._processes)
        sampler.run_nested()
        co_hcn_results.append(sampler.results)


'''
make plots
'''

logz_full = np.array([result.logz[-1] for result in full_results])
logz_h2o = np.array([result.logz[-1] for result in h2o_results])

delta_logz = logz_full - logz_h2o

hist_fig, hist_ax = plt.subplots()
hist_ax.hist(delta_logz)


