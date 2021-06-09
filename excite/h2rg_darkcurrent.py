"""
This is for simulating the dark current of an H2RG 5um cutoff detector
This is specifically for the H2RGs used in the James Webb Space Telescope (JWST)
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal



def diffusion_darkcurrent(T, E_bg=0.6):
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


# generate gaussian noise, at 1Hz, with smooth transitions

# pick length of observation run
observation_duration = 3600  # 1 hour, in seconds
raw_sample_rate = 1000  # in Hz
time = np.linspace(0, observation_duration, num=observation_duration * raw_sample_rate)
# generate noise at 1000 samples per sec
raw_noise = np.random.normal(size=observation_duration * raw_sample_rate)
# filter down to 50 Hz
noise_freq = 50  # in Hz
order = 3  # this is how steep the rolloff is
norm_cutoff = noise_freq / 1000
output = 'sos'  # recommended by documentation
sos = signal.butter(N=order, Wn=norm_cutoff, output=output)

filtered_noise = signal.sosfilt(sos, raw_noise)

# test the rms
print('Raw Noise std =', np.std(raw_noise))
print('Filtered Noise std =', np.std(filtered_noise))
# test plots
stop_time = 600
plt.plot(time[:stop_time], raw_noise[:stop_time])
plt.plot(time[:stop_time], filtered_noise[:stop_time])


