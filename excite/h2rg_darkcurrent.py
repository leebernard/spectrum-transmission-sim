"""
This is for simulating the dark current of an H2RG 5um cutoff detector
This is specifically for the H2RGs used in the James Webb Space Telescope (JWST)
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.stats import poisson

from toolkit import improved_non_uniform_tophat


def diffusion_darkcurrent(T, E_bg=0.6):
    # this is a generic dark current due to diffusion function. 
    # This only incorporates theoretical dark current due to diffusion, not physical
    #E_bg is the bandgap energy, in eV

    # boltzman constant in eV/Kelvin
    k = 8.617333e-5
    return np.exp(-E_bg / (k*T))


def generate_T_noise(time, scale, observation_duration, noise_freq, verbose=False):
    # generate time scale that samples at chosen frequency
    raw_time = np.linspace(0, observation_duration, num=observation_duration * noise_freq)

    # generate noise at chosen freqency
    # choose a scale, increase by 7% to compensate for interpolation smoothing
    scale = scale * 1.07
    raw_noise = np.random.normal(scale=scale,
                                 size=raw_time.size)

    # create interpolation function
    T_noise_func = interp1d(raw_time, raw_noise, kind='cubic')
    # interpolated up to desired sample rate
    T_noise = T_noise_func(time)

    if verbose:
        # test the rms
        print('Raw Noise std =', np.std(raw_noise))
        print('Filtered Noise std =', np.std(T_noise))

    return T_noise


def hpx_threshold(u):
    '''
    Takes a random number between 0-1, and maps it to a logrithmic distribution.

    This follows a scheme where the number of instances below the threshold
    doubles for every factor of 6 increase.
    '''
    tau = 6
    b = .01
    return np.log(u/b) * tau/np.log(2) + 40


def i_dark(T, parameters=(.004, 507.0, 4.6), lambda_co=5.4):
    '''
    This is the empirically derived dark current model from Rauscher et al 2011

    Parameters
    ----------
    T: Temperature in Kelvin
    parameters: the paramters of the fit
    lambda_co: the cutoff wavelength of the H2RG

    Returns
    -------
    The mean dark current of the detector
    '''
    # plank constant times speed of light, in eV um
    hc = 1.239841
    # boltzman constant in eV/Kelvin
    k = 8.617333e-5

    # unpack parameters
    c0 = parameters[0]
    c1 = parameters[1]
    c2 = parameters[2]

    return c0 + c1*np.exp(-hc/(c2*lambda_co) * 1/(k*T))


def nd_i_dark(T, threshholds):

    pass




# generate gaussian noise, at 1Hz, with smooth transitions

# pick length of observation run
# observation_duration = 600  # 10 minutes, in seconds
observation_duration = 60  # one minute
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


T_40, _ = improved_non_uniform_tophat(sample_bins, fine_time,
                                        fine_data=40 + generate_T_noise(fine_time, scale, observation_duration, noise_freq))

T_50, _ = improved_non_uniform_tophat(sample_bins, fine_time,
                                        fine_data=50 + generate_T_noise(fine_time, scale, observation_duration, noise_freq))

T_60, _ = improved_non_uniform_tophat(sample_bins, fine_time,
                                        fine_data=60 + generate_T_noise(fine_time, scale, observation_duration, noise_freq))

T_70, _ = improved_non_uniform_tophat(sample_bins, fine_time,
                                        fine_data=70 + generate_T_noise(fine_time, scale, observation_duration, noise_freq))



# plot the temperature data
time_stop = 300
plt.figure('Temperature_plot')
plt.scatter(sample_time[:time_stop], T_40[:time_stop], label='T=40K')


# generate hot pixel thresholds
thresholds = hpx_threshold(np.random.uniform(size=(2048, 2048)))


# turn temperature variation into a cube
nd_T_40 = np.tile(T_40.astype('float32'), (*thresholds.shape, 1))

# generate dark current values
nd_dark_40 = i_dark(nd_T_40)
nd_htpx_40 = i_dark(nd_T_40, parameters=(.1, 507.0, 4.6))

# produce a binary distribution of hot pixel locations
htpx_map = nd_T_40 >= np.expand_dims(thresholds, 2)


dark_60 = i_dark(T_60)
dark_70 = i_dark(T_70)

# turn hot pixel locations into zeros, then fill them with hot pixel values
simulated_dc_40 = nd_dark_40*np.invert(htpx_map) + nd_htpx_40*htpx_map

plt.figure('pixel plane with hot px', figsize=(8, 8))
plt.imshow(simulated_dc_40[:, :, 0])

# plot the dark current noise
plt.figure('dark current')
plt.plot(sample_time, nd_dark_40.mean(axis=0).mean(axis=0), label='T=40K')
plt.plot(sample_time, simulated_dc_40.mean(axis=0).mean(axis=0), label='T=40K, with hot pixels')
# plt.plot(sample_time, dark_50, label='T=50K')
# plt.plot(sample_time, dark_60, label='T=60k')
# plt.xlim(right=time_stop)
plt.legend()
plt.yscale('log')

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



