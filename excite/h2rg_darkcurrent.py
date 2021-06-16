"""
This is for simulating the dark current of an H2RG 5um cutoff detector
This is specifically for the H2RGs used in the James Webb Space Telescope (JWST)
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from toolkit import instrument_non_uniform_tophat


def diffusion_darkcurrent(T, E_bg=0.6):
    # this is a generic dark current due to diffusion function. 
    # This only incorporates theoretical dark current due to diffusion, not physical
    #E_bg is the bandgap energy, in eV

    # boltzman constant in eV/Kelvin
    k = 8.617333e-5
    return np.exp(-E_bg / (k*T))


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


# generate gaussian noise, at 1Hz, with smooth transitions

# pick length of observation run
observation_duration = 3600  # 1 hour, in seconds
interpolated_sample_rate = 1000  # in Hz
time = np.linspace(0, observation_duration, num=observation_duration * interpolated_sample_rate)

# noise parameters
noise_freq = 50  # in Hz
scale = .005

# generate sampling points
# sampling of data, in Hz
observation_sample_rate = 1
sample_time = np.linspace(0, observation_duration, num=observation_duration * observation_sample_rate)

T_40 = 40 + generate_T_noise(time, scale, observation_duration, noise_freq)
T_50 = 50 + generate_T_noise(time, scale, observation_duration, noise_freq)
T_60 = 60 + generate_T_noise(time, scale, observation_duration, noise_freq)


# plot the temperature data
time_stop = 300
plt.figure('Temperature_plot')
plt.scatter(time[:time_stop], T_40[:time_stop], label='T=40K')
plt.scatter(time[:time_stop], T_50[:time_stop], label='T=50K')
plt.scatter(time[:time_stop], T_60[:time_stop], label='T=60K')
plt.legend()


