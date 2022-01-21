"""
This is for running a delta function through the bf spectrum simulator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import galsim

# three lines: 1000 counts, 10,000 counts, and 50,000 counts

# create numpy array
image_size = (20, 40)
image = np.zeros(image_size)

image[:, 8] = 1000
image[:, 20] = 10000
image[:, 32] = 50000

# convert to galsim image object
galsim_image = galsim.Image(image.copy(), scale=1.0)  # scale is pixel/pixel
# interpolate the image so GalSim can manipulate it
spectrum_interpolated = galsim.InterpolatedImage(galsim_image)
# run the bf simulation
rng = galsim.BaseDeviate(5678)
spectrum_interpolated.drawImage(image=galsim_image,
                                method='phot',
                                # center=(15, 57),
                                sensor=galsim.SiliconSensor(name='lsst_e2v_50_32',
                                                            transpose=True,
                                                            rng=rng,
                                                            diffusion_factor=1.0))

bias = 100
galsim_bf_image = galsim_image.array.copy() + bias
image += bias

fig, ax = plt.subplots(2, 1)
ax[1].imshow(galsim_bf_image, norm=colors.LogNorm(vmin=image.min(), vmax=image.max()), cmap='viridis')

ax[0].imshow(image, norm=colors.LogNorm(vmin=image.min(), vmax=image.max()), cmap='viridis')


