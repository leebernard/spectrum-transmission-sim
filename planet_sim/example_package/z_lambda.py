import numpy as np

import gravity
from constants import k, amu_kg, g_earth, r_earth, r_sun



def z_lambda(sigma_trace, xi, p0, planet_radius, mass, T, planet_mass, sigma_filler=False):
    '''

    Parameters
    ----------
    sigma_trace: array
        Absorption cross section of atmosphere species as a function of wavelength
    p0: float
        Reference pressure of atmosphere; pressure at z=0
    planet_radius: float
        Minimum radius of planet
    mass: float
        mass of trace atomic species
    T: float
        Effective temperature of planet
    planet_mass: float
        mass of the planet in earth masses

    Returns
    -------
    z: float, array
        The amount by which the planet's occultation disk is increased by
        opacity of the atmosphere, as a function of wavelength.
    '''
    # convert planet radius to meters
    r_p = r_earth * planet_radius
    # convert from bars to pa
    pressure = p0 * 100000

    g = gravity(planet_mass, planet_radius)
    h = scale_h(mass, T, g)

    if sigma_filler is not None:
        # calculate average cross section
        sigma = (1 - xi)*sigma_filler + xi*sigma_trace
    else:
        # set volume mixing ratio to 1
        xi = 1
        sigma = sigma_trace

    # set equiv scale height to 1
    tau_eq = 1

    # calculate beta
    beta = pressure / tau_eq * np.sqrt(2*np.pi*r_p)
    return h * np.log(sigma * 1/np.sqrt(k*mass*amu_kg*T*g) * beta)



