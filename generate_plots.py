import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from spectrum_fitter import spectrum_gaussian_fit

# open the files

with fits.open('spectrum_sim_gaussian_bffalse.fits') as hdul:
    galsim_sensor_image = hdul[0].data.copy()

with fits.open('spectrum_sim_gaussian_bftrue.fits') as hdul:
    galsim_bf_image = hdul[0].data.copy()

difference_image = galsim_sensor_image[:, 5:-5] - galsim_bf_image[:, 5:-5]

plt.figure('GalSim simulated spectrum image')

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


plt.figure('mean profile analysis')

plt.subplot(311)
plt.plot(mean_nobf_pixels, mean_sensor_profile)
plt.plot(mean_nobf_pixels, g_nobf(mean_nobf_pixels))
plt.title('Average profile')
plt.ylabel('(flux)')
plt.legend(('Profile', 'Fit'))

plt.subplot(312)
plt.plot(mean_bf_pixels, mean_bf_profile)
plt.plot(mean_bf_pixels, g_withbf(mean_bf_pixels))
plt.title('Profile after bf is applied')
plt.ylabel('(flux)')

plt.subplot(313)
plt.plot(mean_profile_residuals)
plt.title('residuals')
plt.xlabel('(pixels)')
plt.ylabel('(flux)')



plt.show()


