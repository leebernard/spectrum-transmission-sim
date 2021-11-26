import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from spectrum_fitter import spectrum_gaussian_fit
from scipy import constants

from toolkit import spectrum_slicer
fig_size = (8, 12)

# open the files

# title_label = 'Spectrum Simulation with Flux Doubled'
# bffalse_name = 'spectrum_sim_gaussdouble_bffalse.fits'
# bftrue_name = 'spectrum_sim_gaussdouble_bftrue.fits'

# title_label = 'Simulated Spectrum with Gaussian Profiles'
# bffalse_name = 'spectrum_sim_gaussian_bffalse.fits'
# bftrue_name = 'spectrum_sim_gaussian_bftrue.fits'

# title_label = 'Simulated Spectrum, Transposed'
# bffalse_name = 'spectrum_sim_gauss_transpose_bffalse.fits'
# bftrue_name = 'spectrum_sim_gauss_transpose_bftrue.fits'

# title_label = 'Spectrum Simulation with Flux Halved'
# bffalse_name = 'spectrum_sim_gausshalf_bffalse.fits'
# bftrue_name = 'spectrum_sim_gausshalf_bftrue.fits'
#
# title_label = 'Simulated Spectrum using ITL sensor'
# bffalse_name = 'spectrum_sim_gauss_transpose_itl_bffalse.fits'
# bftrue_name = 'spectrum_sim_gauss_transpose_itl_bftrue.fits'

title_label = 'Simulated Spectrum using ChromaStarPy'
bffalse_name = 'spectrum_sim_chromostar_bffalse1.fits'
bftrue_name = 'spectrum_sim_chromostar_bftrue1.fits'

with fits.open(bffalse_name) as hdul:
    galsim_sensor_image = hdul[0].data.copy()

with fits.open(bftrue_name) as hdul:
    galsim_bf_image = hdul[0].data.copy()

difference_image = galsim_sensor_image[:, 5:-5] - galsim_bf_image[:, 5:-5]

plt.figure('GalSim simulated spectrum image', figsize=(8, 8))
plt.suptitle(title_label)

plt.subplot(311)
plt.imshow(galsim_sensor_image[:, 5:-5], cmap='viridis')
plt.title('H-alpha line')
plt.colorbar()

plt.subplot(312)
plt.imshow(galsim_bf_image[:, 5:-5], cmap='viridis')
plt.title('H-alpha line after BF is applied')
plt.colorbar()

plt.subplot(313)
plt.imshow(difference_image, cmap='viridis')
plt.title('Residuals')
plt.colorbar()

# Sanity check: difference image should sum to zero
print('sum of bf vs no bf residuals:', difference_image.sum())
print('Percent difference:', difference_image.sum()/galsim_sensor_image[:, 5:-5].sum() * 100)


'''recreat the spectrum wavelengths'''

filename = '../ast_521/ChromaStarPy/Outputs/BF_sim-5777.0-4.44-0.0-588.0-750.0-Run.sed.txt'

with open(filename, 'r') as file:
    raw_header = file.readline()
    # skip number of wavelength points by printing out
    print(file.readline())
    # skip unit definitions
    print(file.readline())

    # read all the data columns
    raw_data = file.readlines()

    # split the lines into data
    spectrum_data = []
    for x in raw_data:
        spectrum_data.append(x.split())
    # transpose it from row delineated (line format) to column delineated
    spectrum_data = list(map(list, zip(*spectrum_data)))

    # split the data into arrays, converting to floats at the same time
    nanometers = np.array([float(number) for number in spectrum_data[0]])
    log_flux = np.array([float(number) for number in spectrum_data[1]])

spectrum_flux = 10 ** log_flux

# convert from ergs/μm to watts/nm
spectrum_flux = spectrum_flux * 10 ** -7 * 10 ** -3

# convert from flux/m^2 to photons/m^2
c = constants.speed_of_light
h = constants.h

spectrum_counts = spectrum_flux * nanometers / (h * c)

# grab a slice of data
# spectrum_start = 588
# spectrum_end = 591
spectrum_start = 654
spectrum_end = 658
nm_slice, spectrum_counts_slice = spectrum_slicer(spectrum_start,
                                                  spectrum_end,
                                                  nanometers,
                                                  spectrum_counts)
'''End recreate spectrum wavelengths'''

# generate the spectrum via simple summation
sim_nm_per_pixel = .02
number_pixels = int((nm_slice[-1] - nm_slice[0]) / sim_nm_per_pixel)
pixel_grid = np.linspace(nm_slice[0], nm_slice[-1], num=number_pixels)

# make sure to not plot edge effects
trace_ideal = np.sum(galsim_sensor_image, axis=0)
ideal_err = np.sqrt(trace_ideal)
trace_bf = np.sum(galsim_bf_image, axis=0)
bf_err = np.sqrt(trace_bf)
trace_residuals = trace_ideal - trace_bf

spectrace_fig, spect_axs = plt.subplots(2, figsize=(12, 8))
# plt.plot(smeared_spectrum2d[row], label='original data')
spect_axs[0].errorbar(pixel_grid[20:-5], trace_ideal[20:-5], yerr=ideal_err[20:-5], fmt='o', capsize=2.0, label='No charge diffusion')
spect_axs[0].errorbar(pixel_grid[20:-5], trace_bf[20:-5], yerr=bf_err[20:-5], fmt='o', capsize=2.0, label='With charge diffusion')
spect_axs[0].set_title('Spectrum Profile Trace')
# spect_axs[0].set_xlabel('Wavelength (nm)')
spect_axs[0].set_ylabel('Photons (e-)')
spect_axs[0].legend()

resid_err = np.sqrt(bf_err**2 + ideal_err**2)
spect_axs[1].errorbar(pixel_grid[20:-5], trace_residuals[20:-5],  yerr=resid_err[20:-5], fmt='o', capsize=2.0, label='Residuals')
spect_axs[1].axhline(y=0.0, color='r', linestyle='-')
spect_axs[1].set_title('Residuals of above spectrums')
spect_axs[1].set_xlabel('Wavelength (nm)')
spect_axs[1].set_ylabel('Photons (e-)')
spect_axs.legend()


'''Everything below this is spacial profile, and not scientifically interesting'''


# take an average spacial profile
np.std(galsim_sensor_image[14, 15:30])  # center row, relatively flat area in spectrum
np.mean(galsim_sensor_image[14, 15:30])

mean_sensor_profile = np.mean(galsim_sensor_image[:, 15:30], axis=1)
mean_nobf_pixels = np.arange(mean_sensor_profile.size)
mean_bf_profile = np.mean(galsim_bf_image[:, 15:30], axis=1)
mean_bf_pixels = np.arange(mean_bf_profile.size)
mean_profile_residuals = mean_bf_profile - mean_sensor_profile

# fit the profiles
g_nobf, nobf_fitter = spectrum_gaussian_fit(mean_nobf_pixels, mean_sensor_profile, amplitude=50000., mean=14., stddev=1.)
g_withbf, withbf_fitter = spectrum_gaussian_fit(mean_bf_pixels, mean_bf_profile, amplitude=50000., mean=14., stddev=1.)
print('Results:')
print('No bf:', g_nobf.parameters)
print('With bf', g_withbf.parameters)
mean_difference = g_nobf.parameters[1] - g_withbf.parameters[1]
print('Difference in mean:', mean_difference)
bf_growth = (g_withbf.parameters[2] - g_nobf.parameters[2])/g_nobf.parameters[2]
print('Fractional growth in FWHM:', bf_growth)


plt.figure('mean spacial profile analysis', figsize=fig_size)

plt.subplot(311)
plt.plot(mean_nobf_pixels, mean_sensor_profile)
plt.plot(mean_nobf_pixels, g_nobf(mean_nobf_pixels))
plt.title('Average profile in spacial direction, no BF')
plt.ylabel('(flux)')
plt.legend(('Profile', 'Fit'))

plt.subplot(312)
plt.plot(mean_bf_pixels, mean_bf_profile)
plt.plot(mean_bf_pixels, g_withbf(mean_bf_pixels))
plt.title('Profile after BF is applied')
plt.ylabel('(flux)')

plt.subplot(313)
plt.plot(mean_profile_residuals)
plt.plot(g_withbf(mean_bf_pixels) - g_nobf(mean_nobf_pixels))  # residuals of the two fits
plt.title('BF - no-BF')
plt.xlabel('(pixels)')
plt.ylabel('(flux)')
plt.legend(('Profile', 'Fits'))

plt.tight_layout()


# analyze the fit residuals
# sum in quadrature
total1 = np.sqrt(sum(np.square(g_nobf(mean_nobf_pixels) - mean_sensor_profile)))
total2 = np.sqrt(sum(np.square(g_withbf(mean_bf_pixels) - mean_bf_profile)))


# normalize the data sets
norm_g_nobf = g_nobf(mean_nobf_pixels) / sum(g_nobf(mean_nobf_pixels))
norm_data_nobf = mean_sensor_profile / sum(mean_sensor_profile)
no_bf_residuals = np.sqrt(sum(np.square(norm_g_nobf - norm_data_nobf)))

norm_g_yesbf = g_withbf(mean_bf_pixels) / sum(g_withbf(mean_bf_pixels))
norm_data_yesbf = mean_bf_profile / sum(mean_bf_profile)
yes_bf_residuals = np.sqrt(sum(np.square(norm_g_yesbf - norm_data_yesbf)))

print('Sanity check:')
print('ratio of residuals between no bf and with bf')
print('raw summing', total2/total1)
print('summing after normalizing', yes_bf_residuals/no_bf_residuals)

print('Normalized residual values')
print('no BF:', no_bf_residuals)
print('yes BF:', yes_bf_residuals)

plt.figure('Fit Residuals', figsize=fig_size)

plt.subplot(311)
plt.plot(g_nobf(mean_nobf_pixels) - mean_sensor_profile)
plt.title('Residuals of the no-BF fit')
plt.ylabel(r'(flux)')

plt.subplot(312)
plt.plot(g_withbf(mean_bf_pixels) - mean_bf_profile)
plt.title('Residuals of the yes-BF fit')
plt.ylabel(r'(flux)')

plt.subplot(313)
plt.plot(g_withbf(mean_bf_pixels) - g_nobf(mean_nobf_pixels))
plt.title('Residuals of the yes-BF fit minus no-BF fit')
plt.xlabel('(pixels)')
plt.ylabel('(flux)')

plt.tight_layout()

plt.show()


