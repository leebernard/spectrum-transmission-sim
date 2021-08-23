import sys
# import os
print('Python', sys.version)

import numpy as np
import matplotlib.pyplot as plt
import dynesty
import time


from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import minimize
from dynesty import plotting as dyplot
from datetime import datetime

from planet_sim.transit_toolbox import open_cross_section
from planet_sim.transit_toolbox import transit_model_H2OCH4NH3HCN
from planet_sim.transit_toolbox import transit_model_H2OCH4

name = 'macdonald_H2OCH4NH3HCN_R140'
plot = False

start_time = time.time()
print('Starting simulation run on instance', name)
print('Start time:', datetime.now())

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

# na_cross_sections =
# k_cross_sections =
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
# pulled from Wikipedia
rad_planet = 1.35  # in jovian radii
g_planet = 9.4  # m/s
rad_star = 1.203  # solar radii
p0 = 1  # barr
# below is taken from MacDonald 2017
T = 1071  # Kelvin

# log_fna = -5.13
# log_fk =
log_f_h2o = -5.24
log_fch4 = -7.84
log_fnh3 = -6.03
log_fhcn = -6.35
# flip the data to ascending order
flipped_wl = np.flip(fine_wavelengths)
# see Hubble WFC3 slitless spectrograph in NIR
# https://hst-docs.stsci.edu/wfc3ihb/chapter-8-slitless-spectroscopy-with-wfc3/8-1-grism-overview

# this data based upon Deming et al 2013
# resolution of spectrograph
R = 140

# open wavelength sampling of spectrum
sampling_data = './planet_sim/data/HD209458b_demingetal_data'
sampling_wl, sampling_err = open_cross_section(sampling_data)

pixel_delta_wl = np.diff(sampling_wl).mean()
# make pixel bins
# these bins are just a simple mean upsampling
# this is close enough for the purposes of this simulation
wfc3_start = sampling_wl[0] - pixel_delta_wl/2
wfc3_end = sampling_wl[-1] + pixel_delta_wl/2
# pixel_bins = np.linspace(wfc3_start, wfc3_end, sampling_wl.size + 1)

pixel_bins = np.linspace(wfc3_start, wfc3_end, sampling_wl.size*2 + 1)

# pixel_wavelengths = np.linspace(flipped_wl[0], flipped_wl[-1], num=number_pixels)

# pixel_bins = pixel_wavelengths
# test the model generation function
# this produces the 'true' transit spectrum
fixed_parameters = (fine_wavelengths,
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
         log_f_h2o,
         log_fch4,
         log_fnh3,
         log_fhcn)
pixel_wavelengths, pixel_transit_depth = transit_model_H2OCH4NH3HCN(pixel_bins, theta, fixed_parameters)

# generate photon noise from a signal value
# signal = 1.22e9
# noise = (np.random.poisson(lam=signal, size=pixel_transit_depth.size) - signal)/signal

'''
generate noise instances!!!
'''
# convert error from parts per million to fractional
# err = sampling_err*1e-6
err = np.interp(pixel_wavelengths, sampling_wl, sampling_err*1e-6)
# interpolate up to the full res data

num_noise_inst = 100
noise_inst = []
while len(noise_inst) < num_noise_inst:
    # noise_inst.append(np.random.normal(scale=err))
    noise_inst.append(np.random.normal(scale=err))

# add noise to the transit spectrum
noisey_transit_depth = pixel_transit_depth + noise_inst
if plot:
    plt.figure('transit depth R%.2f' %R, figsize=(8, 8))
    plt.subplot(212)
    plt.plot(flipped_wl, np.flip(water_cross_sections)*10**log_f_h2o, label='H2O')
    plt.plot(flipped_wl, np.flip(ch4_cross_sections)*10**log_fch4, label='CH4')
    plt.plot(flipped_wl, np.flip(nh3_cross_sections)*10**log_fnh3, label='NH3')
    plt.plot(flipped_wl, np.flip(hcn_cross_sections)*10**log_fhcn, label='HCN')
    plt.plot(flipped_wl, np.flip(h2_cross_sections), label='H2')
    plt.title('Absorption Cross section')
    plt.legend()
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Cross section (cm^2/molecule)')
    plt.yscale('log')

    plt.subplot(211)
    plt.plot(pixel_wavelengths, pixel_transit_depth, label='Ideal')
    plt.errorbar(pixel_wavelengths, noisey_transit_depth[0], yerr=err, label='Photon noise', fmt='o', capsize=2.0)
    plt.title('Transit depth, R= %d, water= %d ppm' % (R, 10**log_f_h2o/1e-6) )
    plt.legend(('Ideal', 'Photon noise'))
    plt.ylabel('($R_p$/$R_{star}$)$^2$ (%)')
    plt.subplot(211).yaxis.set_major_formatter(FormatStrFormatter('% 1.1e'))



'''Fit the data'''


# define a likelyhood function
def log_likelihood(theta, args):
    # retrieve the global variables
    global pixel_bins
    global transit_data
    global err
    global fixed_parameters
    # only 'y' changes on the fly
    fixed = fixed_parameters
    x = pixel_bins
    yerr = err
    y = args[0]
    _, model = transit_model_H2OCH4NH3HCN(x, theta, fixed)

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

ndim = 6
full_results = []
with Pool() as pool:
    for transit_data in noisey_transit_depth:
        sampler = dynesty.NestedSampler(log_likelihood, prior_trans, ndim,
                                        nlive=500, pool=pool, queue_size=pool._processes, logl_args=[transit_data])
        sampler.run_nested()
        full_results.append(sampler.results)

if plot:
    # make a plot of results
    labels = ["Rad_planet", "T", "log H2O", "log CH4", "log NH3", "log HCN"]
    truths = [rad_planet, T, log_f_h2o, log_fch4, log_fnh3, log_fhcn]
    for result in full_results[:2]:

        fig, axes = dyplot.cornerplot(result, truths=truths, show_titles=True,
                                      title_kwargs={'y': 1.04}, labels=labels,
                                      fig=plt.subplots(len(truths), len(truths), figsize=(10, 10)))
        fig.suptitle('Red lines are true values', fontsize=14)
        # fig.savefig('/test/my_first_cornerplot.png')



# from planet_sim.transit_toolbox import transit_model_H2OCH4


# define a new prior function, with only H2O and CH4
# this is basically the same as only H20
def loglike_h2och4(theta, args):
    # retrieve the global variables
    global pixel_bins
    global transit_data
    global err
    global fixed_parameters
    # only 'y' changes on the fly
    fixed = fixed_parameters
    x = pixel_bins
    y = args[0]
    yerr = err

    _, model = transit_model_H2OCH4(x, theta, fixed)

    sigma = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma + np.log(sigma))


ndim = 4
h2och4_results = []

with Pool() as pool:
    for transit_data in noisey_transit_depth:

        sampler = dynesty.NestedSampler(loglike_h2och4, prior_trans, ndim,
                                        nlive=500, pool=pool, queue_size=pool._processes, logl_args=[transit_data])
        sampler.run_nested()
        h2och4_results.append(sampler.results)

if plot:
    # make a plot of results
    labels = ["Rad_planet", "T", "log H2O", "log CH4"]
    truths = [rad_planet, T, log_f_h2o, log_fch4]
    for result in h2och4_results:

        fig, axes = dyplot.cornerplot(result, truths=truths, show_titles=True,
                                      title_kwargs={'y': 1.04}, labels=labels,
                                      fig=plt.subplots(len(truths), len(truths), figsize=(10, 10)))
        fig.suptitle('Red lines are true values', fontsize=14)
        # fig.savefig('/test/my_first_cornerplot.png')


from dynesty.utils import quantile

# extrat the quantile data
full_qauntiles = []
for results in full_results:
    # extract samples and weights
    samples = results['samples']
    weights = np.exp(results['logwt'] - results['logz'][-1])
    print('Sample shape', samples.shape)

    quantiles = [quantile(x_i, q=[0.025, 0.5, 0.975], weights=weights) for x_i in samples.transpose()]
    full_qauntiles.append(quantiles)


h2och4_quantiles = []
for results in h2och4_results:
    # extract samples and weights
    samples = results['samples']
    weights = np.exp(results['logwt'] - results['logz'][-1])
    quantiles = [quantile(x_i, q=[0.025, 0.5, 0.975], weights=weights) for x_i in samples.transpose()]
    h2och4_quantiles.append(quantiles)


# Extract the evidience
logz_full = np.array([result.logz[-1] for result in full_results])
logz_h2och4 = np.array([result.logz[-1] for result in h2och4_results])

delta_logz = logz_full - logz_h2och4

if plot:
    hist_fig, hist_ax = plt.subplots()
    hist_ax.hist(delta_logz)
    plt.title('H2O-CH4-NH3-HCN vs H2O-CH4, on H2O-CH4 data')
    plt.xlabel('Delta log(z)')

import pickle
import os



# pack the data
full_results_archive = {'noise_data': noise_inst, 'transit_depth':noisey_transit_depth, 'wavelength_bins': pixel_bins,  'H2OCH4NH3HCN_fit': full_results, 'H2OCH4_fit': h2och4_results}
filename = './planet_sim/data/' + name + '_full_retrieval.pkl'
print('Saving to', filename)

s = ''
if os.path.isfile(filename):
    s = input('File already exists. continue...?')

if not s or (s[0] != 'n' and s[0] != 'N'):
    with open(filename, mode='wb') as file:
        pickle.dump(full_results_archive, file)


short_archive = {'noise_data': noise_inst,
                 'logz_full': logz_full,
                 'logz_h2och4': logz_h2och4,
                 'full_quantiles': full_qauntiles,
                 'h2och4_quantiles': h2och4_quantiles
                 }
filename = './planet_sim/data/' + name + '_compact_retrieval'
print('Saving to', filename)
s = ''
if os.path.isfile(filename):
    s = input('File already exists. continue...?')

if not s or (s[0] != 'n' and s[0] != 'N'):
    with open(filename, mode='wb') as file:
        pickle.dump(short_archive, file)


print('Instance', name, 'completed.')
print('End time:', datetime.now())
print('Total runtime: %s seconds' % (time.time() - start_time))







