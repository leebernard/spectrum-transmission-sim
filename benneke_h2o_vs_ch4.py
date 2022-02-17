"""
Charnay: https://ui.adsabs.harvard.edu/abs/2021A%26A...646A.171C/abstract
Bezard (non-gcm): https://ui.adsabs.harvard.edu/abs/2020arXiv201110424B/abstract

Benneke: https://ui.adsabs.harvard.edu/abs/2019ApJ...887L..14B/abstract

Notes:
    Need to get line lists for 600 Kelvin
    Scratch that, 255K
    Make sure to simulate at 0.1 Barr, not 1.0. Talk to Luis about this
"""

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
from planet_sim.transit_toolbox import transit_model_H2O
from planet_sim.transit_toolbox import transit_model_CH4
from planet_sim.transit_toolbox import transit_model_H2OCH4
from planet_sim.transit_toolbox import transit_model_NULL



name = 'benneke_h2o_vs_ch4_1'
# name = 'h2o_true_test'
number_trials = 100
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
# wn_start = 2500  # ~4 um
# 1785 cm^-1 ~= 5.602 um
# wn_end = 10000  # this is 1 um
wn_end = 9302  # 1.075 um

# open these files carefully, because they are potentially over 1Gb in size
water_data_file = './line_lists/1H2-16O_1785-10000_300K_0.100000.sigma'
water_wno, water_cross_sections_raw = open_cross_section(water_data_file, wn_range=(wn_start, wn_end))

ch4_data_file = './line_lists/12C-1H4_1785-10000_300K_0.100000.sigma'
ch4_wno, ch4_cross_sections_raw = open_cross_section(ch4_data_file, wn_range=(wn_start, wn_end))

h2_data_file = './line_lists/H2H2_CIA_300K_0.3bar.txt'
h2_wno, h2_cross_sections_raw = open_cross_section(h2_data_file, wn_range=(wn_start, wn_end))

# interpolate the different cross section grids to the same wavenumber grid
fine_wave_numbers = np.arange(wn_start, wn_end, 3.0)

water_cross_sections = 10**np.interp(fine_wave_numbers, water_wno, np.log10(water_cross_sections_raw))
ch4_cross_sections = 10**np.interp(fine_wave_numbers, ch4_wno, np.log10(ch4_cross_sections_raw))
h2_cross_sections = 10**np.interp(fine_wave_numbers, h2_wno, np.log10(h2_cross_sections_raw))
# h2_cross_sections = None
# convert wavenumber to wavelength in microns
fine_wavelengths = 1e4/fine_wave_numbers
if plot:
    # plot them to check
    plt.figure('compare_cross_section')
    plt.plot(fine_wavelengths, water_cross_sections, label='H2O')
    plt.plot(fine_wavelengths, ch4_cross_sections, label='CH4')
    plt.plot(fine_wavelengths, h2_cross_sections, label='H2')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Cross section (cm^2/molecule)')
    plt.yscale('log')
    plt.legend()

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
# based upon K2-18b
# pulled from Charnay et al 2021
rad_planet = 2.71 / 11.2  # earth radii converted to jovian radii
g_planet = 11.5  # m/s
# charnay_rad_star = 0.411  # actually taken from Bezard et al 2020
benneke_rad_star = 0.45  # solar radii
# use benneke
rad_star = benneke_rad_star
p0 = 0.1  # barr
# approximation of the various literature values
T = 265  # Kelvin

# log_fh20 taken from benneke et al
log_fh2o = -2.08
# log_fch4 taken from benneke et al limit
log_fch4 = -2.60

# flip the data to ascending order
flipped_wl = np.flip(fine_wavelengths)
# see Hubble WFC3 slitless spectrograph in NIR
# https://hst-docs.stsci.edu/wfc3ihb/chapter-8-slitless-spectroscopy-with-wfc3/8-1-grism-overview

# this data based upon Deming et al 2013
# resolution of spectrograph
R = 70

# open wavelength sampling of spectrum
sampling_data = './planet_sim/data/HD209458b_demingetal_data'
sampling_wl, sampling_err = open_cross_section(sampling_data)

pixel_delta_wl = np.diff(sampling_wl).mean()
# generate pixel bins
# these bins are just a simple mean upsampling
# this is close enough for the purposes of this simulation
wfc3_start = sampling_wl[0] - pixel_delta_wl/2
wfc3_end = sampling_wl[-1] + pixel_delta_wl/2
pixel_bins = np.linspace(wfc3_start, wfc3_end, sampling_wl.size + 1)

# pixel_wavelengths = np.linspace(flipped_wl[0], flipped_wl[-1], num=number_pixels)

# pixel_bins = pixel_wavelengths
# test the model generation function
# this produces the 'true' transit spectrum
fixed_h2o = (fine_wavelengths,
                    water_cross_sections,
                    h2_cross_sections,
                    g_planet,
                    rad_star,
                    R)
theta_h2o = (rad_planet,
             T,
             log_fh2o)

fixed_ch4 = (fine_wavelengths,
                    ch4_cross_sections,
                    h2_cross_sections,
                    g_planet,
                    rad_star,
                    R)

theta_ch4 = (rad_planet*1.0,
             T,
             log_fch4)

fixed_h2och4 = (fine_wavelengths,
                    water_cross_sections,
                    ch4_cross_sections,
                    h2_cross_sections,
                    g_planet,
                    rad_star,
                    R)

theta_h2och4 = (rad_planet,
                T,
                log_fh2o,
                log_fch4)

theta_null = (rad_planet,
              T)

fixed_null = (fine_wavelengths,
                    h2_cross_sections,
                    g_planet,
                    rad_star,
                    R)

# generate spectrum
pixel_wavelengths_h2o, pixel_transit_depth_h2o = transit_model_H2O(pixel_bins, theta_h2o, fixed_h2o, p0=p0)

pixel_wavelengths_ch4, pixel_transit_depth_ch4 = transit_model_CH4(pixel_bins, theta_ch4, fixed_ch4, p0=p0)

pixel_wavelengths_null, pixel_transit_depth_null = transit_model_NULL(pixel_bins, theta_null, fixed_null, p0=p0)

if np.array_equal(pixel_wavelengths_h2o, pixel_wavelengths_ch4):
    pixel_wavelengths = pixel_wavelengths_h2o

if plot:
    # compare the spectrums, to check
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pixel_wavelengths_h2o, pixel_transit_depth_h2o * 1e6, 'o', label='H2O')
    ax.plot(pixel_wavelengths_ch4, pixel_transit_depth_ch4 * 1e6, 'o', label='CH4')
    ax.plot(pixel_wavelengths_null, pixel_transit_depth_null * 1e6, 'o', label='H2 only')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('transit depth (ppm)')
    ax.legend()

'''
generate noise instances!!!
'''


# convert error from parts per million to fractional
err = sampling_err*1e-6

num_noise_inst = number_trials
noise_inst = []
while len(noise_inst) < num_noise_inst:
    # increase noise by 0%
    noise_inst.append(np.random.normal(scale=err*1))

# add noise to the h2o transit spectrum
noisey_transit_depth_h2o = pixel_transit_depth_h2o + noise_inst

# add noise to the ch4 transit spectrum
noisey_transit_depth_ch4 = pixel_transit_depth_h2o + noise_inst
if plot:
    plt.figure('transit depth R%.2f' %R, figsize=(8, 8))
    plt.subplot(212)
    plt.plot(flipped_wl, np.flip(water_cross_sections)*10**log_fh2o, label='H2O')
    plt.plot(flipped_wl, np.flip(ch4_cross_sections)*10**log_fch4, label='CH4')
    plt.plot(flipped_wl, np.flip(h2_cross_sections), label='H2')
    plt.title('Absorption Cross section')
    plt.legend()
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Cross section (cm^2/molecule)')
    plt.yscale('log')

    plt.subplot(211)
    plt.plot(pixel_wavelengths, pixel_transit_depth_h2o, label='Ideal')
    plt.errorbar(pixel_wavelengths, noisey_transit_depth_h2o[0], yerr=err, label='Photon noise', fmt='o', capsize=2.0)
    plt.title('Transit depth, R= %d, water= %d ppm' % (R, 10**log_fh2o/1e-6) )
    plt.legend(('Ideal', 'Photon noise'))
    plt.ylabel('($R_p$/$R_{star}$)$^2$ (%)')
    plt.subplot(211).yaxis.set_major_formatter(FormatStrFormatter('% 1.1e'))



'''Fit the data'''

# define a likelyhood function
def log_likelihood_h2o(theta, y, fixed_parameters):
    # retrieve the global variables

    global pixel_bins
    global transit_data
    global err
    # only 'y' changes on the fly
    x = pixel_bins
    yerr = err
    _, model = transit_model_H2O(x, theta, fixed_parameters, p0=p0)

    sigma = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma + np.log(sigma))


# define a prior function
def prior_trans(u):
    # u is random samples from the unit cube
    x = np.array(u)  # copy u

    # planet radius prior
    x[0] = u[0]*10
    # Temperature
    x[1] = u[1]*(3000-100) + 100

    # set the trace species to uniform priors
    x[2:] = u[2:]*11 - 11.1

    # global print_number
    # if print_number < 100:
    #     print_number += 1
    #     print('parameter values:', x)
    return x

# plot = True

from multiprocessing import Pool

ndim = len(theta_h2o)
h2o_results_h2otrue = []
with Pool() as pool:
    for transit_data in noisey_transit_depth_h2o:
        sampler = dynesty.NestedSampler(log_likelihood_h2o, prior_trans, ndim,
                                        nlive=500, pool=pool, queue_size=pool._processes, logl_args=(transit_data, fixed_h2o))
        sampler.run_nested()
        h2o_results_h2otrue.append(sampler.results)


# again, but for the ch4 spectrum
h2o_results_ch4true = []
with Pool() as pool:
    for transit_data in noisey_transit_depth_ch4:
        sampler = dynesty.NestedSampler(log_likelihood_h2o, prior_trans, ndim,
                                        nlive=500, pool=pool, queue_size=pool._processes, logl_args=(transit_data, fixed_h2o))
        sampler.run_nested()
        h2o_results_ch4true.append(sampler.results)



if plot:
    # make a plot of results
    labels = ["Rad_planet", "T", "log H2O"]
    truths = [rad_planet, T, log_fh2o]
    for result in h2o_results_h2otrue:

        fig, axes = dyplot.cornerplot(result, truths=truths, show_titles=True,
                                      title_kwargs={'y': 1.04}, labels=labels,
                                      fig=plt.subplots(len(truths), len(truths), figsize=(10, 10)))
        fig.suptitle('Red lines are true values', fontsize=14)
        # fig.savefig('/test/my_first_cornerplot.png')



# from planet_sim.transit_toolbox import transit_model_H2OCH4


# define a new prior function, with only H2O and CH4
# this is basically the same as only H20
def loglike_ch4(theta, y, fixed_parameters):
    # retrieve the global variables
    global pixel_bins
    global transit_data
    global err
    # only 'y' changes on the fly
    fixed = fixed_parameters
    x = pixel_bins
    yerr = err

    _, model = transit_model_CH4(x, theta, fixed, p0=p0)

    sigma = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma + np.log(sigma))


ndim = len(theta_ch4)
ch4_results_h2otrue = []
with Pool() as pool:
    for transit_data in noisey_transit_depth_h2o:

        sampler = dynesty.NestedSampler(loglike_ch4, prior_trans, ndim,
                                        nlive=500, pool=pool, queue_size=pool._processes, logl_args=(transit_data, fixed_ch4))
        sampler.run_nested()
        ch4_results_h2otrue.append(sampler.results)

# again, for the ch4 case
ch4_results_ch4true = []
with Pool() as pool:
    for transit_data in noisey_transit_depth_ch4:

        sampler = dynesty.NestedSampler(loglike_ch4, prior_trans, ndim,
                                        nlive=500, pool=pool, queue_size=pool._processes, logl_args=(transit_data, fixed_ch4))
        sampler.run_nested()
        ch4_results_ch4true.append(sampler.results)

if plot:
    # make a plot of results
    labels = ["Rad_planet", "T", "log CH4"]
    truths = [rad_planet, T, log_fch4]
    for result in ch4_results_h2otrue:

        fig, axes = dyplot.cornerplot(result, truths=truths, show_titles=True,
                                      title_kwargs={'y': 1.04}, labels=labels,
                                      fig=plt.subplots(len(truths), len(truths), figsize=(10, 10)))
        fig.suptitle('Red lines are true values', fontsize=14)
        # fig.savefig('/test/my_first_cornerplot.png')


'''Extract relevant results from the analysis'''

from dynesty.utils import quantile

# extract the quantile data
h2o_qauntiles_h2otrue = []
for results in h2o_results_h2otrue:
    # extract samples and weights
    samples = results['samples']
    weights = np.exp(results['logwt'] - results['logz'][-1])
    print('Sample shape', samples.shape)

    quantiles = [quantile(x_i, q=[0.025, 0.5, 0.975], weights=weights) for x_i in samples.transpose()]
    h2o_qauntiles_h2otrue.append(quantiles)

h2o_qauntiles_ch4true = []
for results in h2o_results_ch4true:
    # extract samples and weights
    samples = results['samples']
    weights = np.exp(results['logwt'] - results['logz'][-1])
    print('Sample shape', samples.shape)

    quantiles = [quantile(x_i, q=[0.025, 0.5, 0.975], weights=weights) for x_i in samples.transpose()]
    h2o_qauntiles_ch4true.append(quantiles)


ch4_quantiles_h2otrue = []
for results in ch4_results_h2otrue:
    # extract samples and weights
    samples = results['samples']
    weights = np.exp(results['logwt'] - results['logz'][-1])
    quantiles = [quantile(x_i, q=[0.025, 0.5, 0.975], weights=weights) for x_i in samples.transpose()]
    ch4_quantiles_h2otrue.append(quantiles)


ch4_quantiles_ch4true = []
for results in ch4_results_ch4true:
    # extract samples and weights
    samples = results['samples']
    weights = np.exp(results['logwt'] - results['logz'][-1])
    quantiles = [quantile(x_i, q=[0.025, 0.5, 0.975], weights=weights) for x_i in samples.transpose()]
    ch4_quantiles_ch4true.append(quantiles)


'''Analyze the results'''

# Extract the evidience
logz_h2o_h2otrue = np.array([result.logz[-1] for result in h2o_results_h2otrue])
logz_ch4_h2otrue = np.array([result.logz[-1] for result in ch4_results_h2otrue])

delta_logz_h2otrue = logz_h2o_h2otrue - logz_ch4_h2otrue


# again, for the ch4 case
logz_h2o_ch4true = np.array([result.logz[-1] for result in h2o_results_ch4true])
logz_ch4_ch4true = np.array([result.logz[-1] for result in ch4_results_ch4true])

delta_logz_ch4true = logz_h2o_ch4true - logz_ch4_ch4true

if plot:
    hist_fig, hist_ax = plt.subplots()
    hist_ax.hist(delta_logz_h2otrue)
    plt.title('H2O-CH4-NH3-HCN vs H2O-CH4, on H2O-CH4 data')
    plt.xlabel('Delta log(z)')


'''Save the results'''

import pickle
import os



# pack the data
h2otrue_full_results =  {'noise_data': noise_inst,
                        'transit_depth':noisey_transit_depth_h2o,
                        'free_param_values': theta_h2o,
                        'wavelength_bins': pixel_bins,
                        'H2O_fit': h2o_results_h2otrue,
                        'CH4_fit': ch4_results_h2otrue}

ch4true_full_results =  {'noise_data': noise_inst,
                        'transit_depth':noisey_transit_depth_ch4,
                        'free_param_values': theta_ch4,
                        'wavelength_bins': pixel_bins,
                        'H2O_fit': h2o_results_ch4true,
                        'CH4_fit': ch4_results_ch4true}

full_results_archive = {'h2o_true': h2otrue_full_results,
                        'ch4_true': ch4true_full_results}
filename = './planet_sim/data/' + name + '_full_retrieval.pkl'
print('Saving to', filename)

n=0
new_filename = filename
while os.path.isfile(new_filename):
    n+=1
    print('File already exists.')
    index = filename.find('.pkl')
    new_filename = filename[:index] + '_%d' %n + filename[index:]
    print('Try again, saving to', new_filename)

filename = new_filename
with open(filename, mode='wb') as file:
    pickle.dump(full_results_archive, file)
    print('Saved to', file)


short_results_h2otrue = {'noise_data': noise_inst,
                 'free_param_values': theta_h2o,
                 'logz_h2o': logz_h2o_h2otrue,
                 'logz_ch4': logz_ch4_h2otrue,
                 'h2o_quantiles': h2o_qauntiles_h2otrue,
                 'ch4_quantiles': ch4_quantiles_h2otrue
                 }

short_results_ch4true = {'noise_data': noise_inst,
                 'free_param_values': theta_ch4,
                 'logz_h2o': logz_h2o_ch4true,
                 'logz_ch4': logz_ch4_ch4true,
                 'h2o_quantiles': h2o_qauntiles_ch4true,
                 'ch4_quantiles': ch4_quantiles_ch4true
                 }

short_archive = {'h2o_true': short_results_h2otrue,
                 'ch4_true': short_results_ch4true}

filename = './planet_sim/data/' + name + '_compact_retrieval.pkl'
print('Saving to', filename)

n=0
new_filename = filename
while os.path.isfile(new_filename):
    n+=1
    print('File already exists.')
    index = filename.find('.pkl')
    new_filename = filename[:index] + '_%d' %n + filename[index:]
    print('Try again, saving to', new_filename)

filename = new_filename
with open(filename, mode='wb') as file:
    pickle.dump(short_archive, file)
    print('Saved to', file)


print('Instance', name, 'completed.')
print('End time:', datetime.now())
print('Total runtime: %s seconds' % (time.time() - start_time))











