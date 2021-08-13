"""
This is for simulating the dark current of an H2RG 5um cutoff detector
This is specifically for the H2RGs used in the James Webb Space Telescope (JWST)
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.stats import poisson

from toolkit import improved_non_uniform_tophat

from excite.h2rg_toolbox import generate_T_noise
from excite.h2rg_toolbox import generate_dc_means
from excite.h2rg_toolbox import hpx_threshold



# generate gaussian noise, at 1Hz, with smooth transitions

# pick length of observation run
# observation_duration = 600  # 10 minutes, in seconds
observation_duration = 300  # 10 minutes
interpolated_sample_rate = 1000  # in Hz
fine_time = np.linspace(0, observation_duration, num=observation_duration * interpolated_sample_rate)

# noise parameters
noise_freq = 50  # in Hz
# scale = .005
scale = .050  # 50mK

full_well = 70000
# generate sampling points
# sampling of data, in Hz
observation_sample_rate = 1
sample_bins = np.linspace(0, observation_duration, num=observation_duration * observation_sample_rate)
sample_time = sample_bins[1:]

temp_curve = np.linspace(40, 90, num=10)
# generate temperature data cube
temp_data_cube = []
for temp_value in temp_curve:
    temp_data, _ = improved_non_uniform_tophat(sample_bins, fine_time,
                                                    fine_data=temp_value + generate_T_noise(fine_time, scale, observation_duration, noise_freq))
    temp_data_cube.append(temp_data)


# generate hot pixel thresholds
thresholds = hpx_threshold(np.random.uniform(size=(512, 1024)))


# # plot the dark current mean values
# plt.figure('dark current')
# plt.plot(sample_time, nd_dark_40.mean(axis=0).mean(axis=0), label='T=40K')
# plt.plot(sample_time, dc_40_rates.mean(axis=0).mean(axis=0), label='T=40K, with hot pixels')
# # plt.plot(sample_time, dark_50, label='T=50K')
# # plt.plot(sample_time, dark_60, label='T=60k')
# # plt.xlim(right=time_stop)
# plt.legend()
# plt.yscale('log')


# generate actual dark current values
# sum up the cube over the time axis
T_40 = temp_data_cube[0]
T_67 = temp_data_cube[5]
dc_67_means = generate_dc_means(T_67, thresholds)
simulated_dc_67 = poisson.rvs(dc_67_means)
simulated_dc_rates = simulated_dc_67/observation_duration

# show the dark current
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
pcm = ax[0].imshow(simulated_dc_67)
fig.colorbar(pcm, ax=ax[0], fraction=0.046, pad=0.04)

bins = np.linspace(0, simulated_dc_67.max()/observation_duration, simulated_dc_67.max())
ax[1].hist(simulated_dc_67.flatten()/observation_duration, bins=bins)
ax[1].set_xlabel('Dark Current (e-) for T=%.1f'  %temp_curve[5])
fig.suptitle('Exposure of %.2f minutes'  %(observation_duration/60))


bins = np.linspace(0, 1.2, 350)
fig, ax = plt.subplots(len(temp_data_cube), 2, figsize=(8, 12))
for n, temp_data in enumerate(temp_data_cube):
    print('Temperature %.2f'  %temp_curve[n])
    dc_means = generate_dc_means(temp_data, thresholds)
    simulated_dc = poisson.rvs(dc_means)
    pcm = ax[n, 0].imshow(simulated_dc/observation_duration)
    fig.colorbar(pcm, ax=ax[n, 0], fraction=0.046, pad=0.04)
    # bins = np.linspace(0, simulated_dc.max()/observation_duration, simulated_dc.max())

    ax[n, 1].hist(simulated_dc.flatten()/observation_duration, bins=bins)
    ax[n, 1].set_xlabel('Dark Current (e-/s), T=%.1f' % temp_curve[n])
    ax[n, 1].set_xlim
# fig.suptitle('Exposure of %.2f minutes'  %(observation_duration/60))
plt.tight_layout()
plt.savefig('test.png')


'''
# generate a cube of data, for 40-90 degrees, every 2 degrees
temp_curve = np.linspace(40, 90, num=25)

observation_duration = 1800  # 1/2 hour, in seconds
interpolated_sample_rate = 1000  # in Hz
fine_time = np.linspace(0, observation_duration, num=observation_duration * interpolated_sample_rate)

# noise parameters
noise_freq = 50  # in Hz
scale = scale

# generate sampling points
# sampling of data, in Hz
observation_sample_rate = 1
sample_bins = np.linspace(0, observation_duration, num=observation_duration * observation_sample_rate)
sample_time = sample_bins[1:]

# generate temperature data cube
temp_data_cube = []
for temp_value in temp_curve:
    temp_data, _ = improved_non_uniform_tophat(sample_bins, fine_time,
                                                    fine_data=temp_value + generate_T_noise(fine_time, scale, observation_duration, noise_freq))
    temp_data_cube.append(temp_data)

temp_data_cube = np.array(temp_data_cube)

# generate dark current data cube
dc_rate_cube = i_dark(temp_data_cube)



stddev_dc_rate = np.std(dc_rate_cube, axis=1)
# normalize by full well, and convert to ppm
dc_noise = stddev_dc_rate/full_well * 1e6


# need to compare the above noise to dc poisson noise
integration_time = 30  # in secs
start = 0
stop = integration_time * observation_sample_rate

# generate data with poisson noise
# this produces single pixel data, need to expand to a pixel array
# dc_30s_cube = poisson.rvs(np.sum(dc_rate_cube[:, start:stop], axis=1))

# generate dark current for 30 secs
dc_30s = np.sum(dc_rate_cube[:, start:stop], axis=1)
# take the 'photon' noise limit as the inherent noise
# also convert to ppm of full well
photon_noise_30s = np.sqrt(dc_30s)/full_well * 1e6

plt.figure('dark current sn')
plt.scatter(temp_curve, dc_noise, label='noise due to 50Hz 50mK jitter')
plt.scatter(temp_curve, photon_noise_30s, label='poisson noise at 30s')
plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel('Noise in ppm of full well')
plt.yscale('log')
plt.ylim(bottom=1e-5)
'''


