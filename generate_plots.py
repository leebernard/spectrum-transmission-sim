import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

# open the files

with fits.open('spectrum_sim_gaussian_bffalse2.fits') as hdul:
    galsim_sensor_image = hdul[0].data.copy()

with fits.open('spectrum_sim_gaussian_bftrue2.fits') as hdul:
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
mean_bf_profile = np.mean(galsim_bf_image[:, 15:30], axis=1)
mean_profile_residuals = mean_bf_profile - mean_sensor_profile


plt.figure('mean profile analysis')
plt.subplot(311)
plt.plot(mean_sensor_profile)
plt.title('Average profile')
plt.ylabel('(flux)')
plt.subplot(312)
plt.plot(mean_bf_profile)
plt.title('Profile after bf is applied')
plt.ylabel('(flux)')
plt.subplot(313)
plt.plot(mean_profile_residuals)
plt.title('residuals')
plt.xlabel('(pixels)')
plt.ylabel('(flux)')



plt.show()


