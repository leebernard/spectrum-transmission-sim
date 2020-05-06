import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from spectrum_fitter import spectrum_gaussian_fit

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

title_label = 'Spectrum Simulation with Flux Halved'
bffalse_name = 'spectrum_sim_gausshalf_bffalse.fits'
bftrue_name = 'spectrum_sim_gausshalf_bftrue.fits'
#
# title_label = 'Simulated Spectrum using ITL sensor'
# bffalse_name = 'spectrum_sim_gauss_transpose_itl_bffalse.fits'
# bftrue_name = 'spectrum_sim_gauss_transpose_itl_bftrue.fits'

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


# take an average profile
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


plt.figure('mean profile analysis', figsize=fig_size)

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
# sum in quadtrature
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


