import matplotlib.pyplot as plt
import numpy as np
import os

from astropy.io import fits
# from spectrum_fitter import spectrum_gaussian_fit

script_dir = os.path.dirname(__file__) # <-- absolute dir the script is in

title_label = 'Gaussian Spot Image'
bffalse_name = os.path.join(script_dir, r'output/spot_nobf.fits')
bftrue_name = os.path.join(script_dir, r'output/spot_lsst_e2v_50_32_bf_1.fits')
bf_itl_name = os.path.join(script_dir, r'output/spot_lsst_itl_50_32_bf_1.fits')

with fits.open(bffalse_name) as hdul:
    galsim_sensor_image = hdul[0].data.copy()

with fits.open(bftrue_name) as hdul:
    galsim_bf_image = hdul[0].data.copy()

with fits.open(bf_itl_name) as hdul:
    galsim_itl_image = hdul[0].data.copy()

difference_image = galsim_bf_image - galsim_sensor_image
itl_residuals = galsim_itl_image - galsim_sensor_image

fig_size = (9, 8)
plt.figure('GalSim Gaussian spot image', figsize=fig_size)
plt.suptitle(title_label)

plt.subplot(221)
plt.imshow(galsim_sensor_image, cmap='viridis')
plt.title('Gaussian Spot, Ideal Sensor')
plt.colorbar()

plt.subplot(222)
plt.imshow(galsim_bf_image, cmap='viridis')
plt.title('Gaussian Spot, E2V sensor, 50 update')
plt.colorbar()

plt.subplot(223)
plt.imshow(difference_image, cmap='viridis', vmax=-difference_image.min())
plt.title('Residuals, E2V simulation, 50 update')
plt.colorbar()

plt.subplot(224)
plt.imshow(itl_residuals, cmap='viridis', vmax=-itl_residuals.min())
plt.title('Resduals, ITL simulation, 50 update')
plt.colorbar()

plt.show()
