import numpy as np

from scipy.interpolate import interp1d

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


def generate_dc_rates(T_data, hpx_thresholds):

    # turn temperature variation into a cube
    nd_T_data = np.tile(T_data.astype('float32'), (*hpx_thresholds.shape, 1))

    # generate dark current mean values
    nd_dark_40 = i_dark(nd_T_data)
    nd_htpx_40 = i_dark(nd_T_data, parameters=(.06, 507.0, 4.6))

    # produce a binary distribution of hot pixel locations
    htpx_map = nd_T_data >= np.expand_dims(hpx_thresholds, 2)

    return nd_dark_40*np.invert(htpx_map) + nd_htpx_40*htpx_map


def generate_dc_means(T_data, hpx_thresholds):
    dc_rates = generate_dc_rates(T_data, hpx_thresholds)

    # sum along the time axis
    return np.sum(dc_rates, axis=2)


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