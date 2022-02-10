"""
Functions for running the planet transit simulation
"""

import numpy as np
import warnings
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# import matplotlib.pyplot as plt
from toolkit import spectrum_slicer
from toolkit import instrument_non_uniform_tophat
from toolkit import improved_non_uniform_tophat
from toolkit import consecutive_mean


# global constants
k = 1.38e-23  # boltzmann constant k_b in J/K
amu_kg = 1.66e-27  # kg/amu
g_jovian = 24.79  # m/s^2
r_jovian = 7.1492e7  # meters
r_sun = 6.957e8  # meters

# masses of species
mass_h2 = 2.3  # mean molecular weight of H2 He mix
# mass_na = 11
# mass_k = 19
mass_water = 18
mass_ch4 = 12 + 1*4
mass_nh3 = 14 + 1*3
mass_hcn = 1+12+14
mass_co = 12+16
mass_hcn = 1+12+14


def open_cross_section(filename, wn_range=None, verbose=False, skiplines=None):
    with open(filename) as file:
        if skiplines:
            if verbose: print('skiping %d lines' % skiplines)
            raw_data = file.readlines()[skiplines:]
        else:
            raw_data = file.readlines()

        wave_numbers = []
        cross_sections = []

        if verbose:
            print('raw')

        for x in raw_data:
            if x == '\n':
                warnings.warn('Cross section read in terminated early due to empty line!', UserWarning)
                break
            else:
                wave_string, cross_string = x.split()
                wave_numbers.append(float(wave_string))
                cross_sections.append(float(cross_string))
        wave_numbers = np.array(wave_numbers)
        cross_sections = np.array(cross_sections)

    if wn_range is None:
        return wave_numbers, cross_sections
    else:
        # wn range needs to be exactly 2 values
        # explicitly pass those two values
        wn_start, wn_end = wn_range
        return spectrum_slicer(wn_start, wn_end, wave_numbers, cross_sections)


def gravity(mass_planet, rad_planet):
    return g_jovian * mass_planet / (rad_planet ** 2)


def scale_h(mass, T, g):
    '''
    Calculates the scale height

    Parameters
    ----------
    mass: float
        average mass of atmosphere species, in atomic units
    T: float
        Average temperature of atmosphere
    g: float
        Gravitational constant

    Returns
    -------
    The scale height of an atmosphere
    '''
    return k*T/(mass*amu_kg * g)


def z_lambda(sigma_trace, xi, p0, planet_radius, mass, T, g, sigma_filler=None):
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
    g: float
        gravity of the planet, in m/s
    sigma_filler: float
        cross section of the filler gas

    Returns
    -------
    z: float, array
        The amount by which the planet's occultation disk is increased by
        opacity of the atmosphere, as a function of wavelength.
    '''
    # convert planet radius to meters
    r_p = r_jovian * planet_radius
    # convert from bars to pa
    pressure = p0 * 100000

    # g = gravity(planet_mass, planet_radius)
    h = scale_h(mass, T, g)

    if sigma_filler is not None:
        # calculate average cross section
        sigma = (1 - np.sum(xi))*sigma_filler + np.sum(xi*sigma_trace, axis=0)
    else:

        sigma = np.sum(xi*sigma_trace, axis=0)

    # set equiv scale height to 0.56 (Line and Parmenteir 2016)
    tau_eq = 0.56

    # calculate beta
    beta = pressure / tau_eq * np.sqrt(2*np.pi*r_p)
    return h * np.log(sigma * 1/np.sqrt(k*mass*amu_kg*T*g) * beta)


def alpha_lambda(sigma_trace, xi, planet_radius, p0, T, mass, g, star_radius, sigma_filler=None):
    '''

    Parameters
    ----------
    sigma
    planet_radius
    p0
    T
    mass
    planet_mass
    star_radius

    Returns
    -------
    The eclipse depth as a function of wavelength
    '''
    # convert to meters
    r_planet = r_jovian * planet_radius
    r_star = r_sun * star_radius

    z = z_lambda(sigma_trace=sigma_trace,
                 xi=xi,
                 p0=p0,
                 planet_radius=planet_radius,
                 mass=mass,
                 T=T,
                 g=g,
                 sigma_filler=sigma_filler)

    return (r_planet / r_star)**2 + (2 * r_planet * z)/(r_star**2)


def gen_measured_transit(R, pixel_bins, fine_wl, fine_transit):
    # filter the spectrum slice with a gaussian
    # the average resolution of the spectrum in micrometers. This corresponds to FWHM of spectrum lines
    resolution = np.mean(fine_wl)/R
    fwhm = 1/np.mean(np.diff(fine_wl)) * resolution  # the fwhm in terms of data spacing
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    filtered_transit = gaussian_filter(fine_transit.data, sigma)

    # sample the spectrum into bins
    # this represents pixels
    # check the inputs
    # print('input sizes', pixel_bins.size, fine_wl.size, filtered_transit.size)
    pixel_transit_depth, _ = improved_non_uniform_tophat(pixel_bins, fine_wl, filtered_transit)

    return consecutive_mean(pixel_bins), pixel_transit_depth


def transit_spectra_model(pixel_wavelengths, theta, fixed):
    # fixed global variables
    p0 = 1
    global mass_h2  # mean molecular weight of H2 He mix
    global mass_water
    global mass_co
    global mass_hcn

    # unpack model variables
    rad_planet, T, log_f_h2o, log_fco, log_fhcn = theta

    # unpack log ratios
    water_ratio = 10**log_f_h2o
    co_ratio = 10**log_fco
    hcn_ratio = 10**log_fhcn

    # package the ratios into a summable list
    trace_ratios = np.array([
        [water_ratio],
        [co_ratio],
        [hcn_ratio]
    ])

    # print('trace ratios', trace_ratios)
    # unpack known priors
    fine_wavelengths, h2o_cross_sections, co_cross_sections, hcn_cross_sections, h2_cross_sections, g_planet, rad_star, R = fixed

    # package the cross sections into an array
    sigma_trace = np.array(
        [h2o_cross_sections,
         co_cross_sections,
         hcn_cross_sections]
    )


    # he/h2 ratio is disabled, since I don't have the cross-sections
    # h2he_ratio = .17
    # mass_h2he = (1 - h2he_ratio) * 2 + h2he_ratio * 4
    # mass = (1 - water_ratio) * mass_h2he + water_ratio * mass_water

    # temporary mass cause I don't have He cross sections yet
    sum_ratios = water_ratio + co_ratio + hcn_ratio
    mass = (1 - sum_ratios)*mass_h2 + water_ratio*mass_water + co_ratio*mass_co + hcn_ratio*mass_hcn

    transit_depth = alpha_lambda(sigma_trace=sigma_trace,
                                 xi=trace_ratios,
                                 planet_radius=rad_planet,
                                 p0=p0,
                                 T=T,
                                 mass=mass,
                                 g=g_planet,
                                 star_radius=rad_star,
                                 sigma_filler=h2_cross_sections
                                 )

    '''Sample the data into a spectrum'''
    # set the resolution of the spectrometer
    # flip the data to ascending order
    fine_wavelengths = np.flip(fine_wavelengths)
    transit_depth = np.flip(transit_depth)

    # interpolate the data to even spacing
    # this should be unnecessary
    fine_um_grid = np.linspace(fine_wavelengths[0], fine_wavelengths[-1], num=fine_wavelengths.size)
    fine_transit_grid = griddata(fine_wavelengths, transit_depth, xi=fine_um_grid, method='linear')

    out_wavelengths, pixel_transit_depth = gen_measured_transit(R=R, pixel_bins=pixel_wavelengths,
                                                                fine_wl=fine_um_grid,
                                                                fine_transit=fine_transit_grid)
    '''end sample data into spectrum'''

    return out_wavelengths, pixel_transit_depth


def transit_spectra_test(pixel_wavelengths, theta, fixed, p0=1, debug=False):

    """
    test routine using h2o atmosphere. Has the option of using CIA

    """
    # fixed global variables
    global mass_h2  # mean molecular weight of H2 He mix
    global mass_water
    global mass_co
    global mass_hcn

    rad_planet, T, log_f_h2o = theta

    # unpack log ratios
    water_ratio = 10**log_f_h2o

    # package the ratios into a summable list
    trace_ratios = np.atleast_2d(water_ratio)

    # unpack known priors
    fine_wavelengths, h2o_cross_sections, h2_cross_sections, g_planet, rad_star, R = fixed

    # package the cross sections into an array
    sigma_trace = np.atleast_2d(h2o_cross_sections)

    if h2_cross_sections is not None:
        # if filler gas is provided, calculate the mean molecular weight
        weighted_mass_f = mass_water * water_ratio
        mass = (1 - water_ratio)*mass_h2 + weighted_mass_f
        if debug:
            print('Molecular weight calculated', mass)
    else:
        # fix atmosphere mass to 2.3
        mass = 2.3
        if debug:
            print('Molecular weight fixed to', mass)

    # generate the 'true' transit depth
    transit_depth = alpha_lambda(sigma_trace=sigma_trace,
                                 xi=trace_ratios,
                                 planet_radius=rad_planet,
                                 p0=p0,
                                 T=T,
                                 mass=mass,
                                 g=g_planet,
                                 star_radius=rad_star,
                                 sigma_filler=h2_cross_sections
                                 )

    # test print
    if debug:
        print('transit_depth', transit_depth.shape)

    '''Sample the data into a spectrum'''
    # set the resolution of the spectrometer
    # flip the data to ascending order
    fine_wavelengths = np.flip(fine_wavelengths)
    transit_depth = np.flip(transit_depth)

    out_wavelengths, pixel_transit_depth = gen_measured_transit(R=R, pixel_bins=pixel_wavelengths,
                                                                fine_wl=fine_wavelengths,
                                                                fine_transit=transit_depth)
    '''end sample data into spectrum'''

    return out_wavelengths, pixel_transit_depth


def transit_spectra_h2o_only(pixel_wavelengths, theta, fixed, p0=1):
    """
    Old water only atmosphere routine. Superceeded by transit_spectra_test

    Parameters
    ----------
    p0
    pixel_wavelengths
    theta
    fixed

    Returns
    -------

    """
    # fixed global variables
    global mass_h2  # mean molecular weight of H2 He mix
    global mass_water
    global mass_co
    global mass_hcn

    # unpack model variables
    rad_planet, T, log_f_h2o = theta

    # unpack log ratios
    water_ratio = 10**log_f_h2o

    # package the ratios into a summable list
    trace_ratios = np.array([
        [water_ratio]
    ])

    # print('trace ratios', trace_ratios)
    # unpack known priors
    fine_wavelengths, h2o_cross_sections, co_cross_sections, hcn_cross_sections, h2_cross_sections, g_planet, rad_star, R = fixed

    # package the cross sections into an array
    sigma_trace = np.array(
        [h2o_cross_sections]
    )

    # he/h2 ratio is disabled, since I don't have the cross-sections
    # h2he_ratio = .17
    # mass_h2he = (1 - h2he_ratio) * 2 + h2he_ratio * 4
    # mass = (1 - water_ratio) * mass_h2he + water_ratio * mass_water

    # temporary mass cause I don't have He cross sections yet
    sum_ratios = water_ratio
    mass = (1 - sum_ratios)*mass_h2 + water_ratio*mass_water

    # generate the 'true' transit depth
    transit_depth = alpha_lambda(sigma_trace=sigma_trace,
                                 xi=trace_ratios,
                                 planet_radius=rad_planet,
                                 p0=p0,
                                 T=T,
                                 mass=mass,
                                 g=g_planet,
                                 star_radius=rad_star,
                                 sigma_filler=h2_cross_sections
                                 )

    '''Sample the transit depth into spectrum data'''
    # set the resolution of the spectrometer
    # flip the data to ascending order
    fine_wavelengths = np.flip(fine_wavelengths)
    transit_depth = np.flip(transit_depth)

    # interpolate the data to even spacing
    fine_um_grid = np.linspace(fine_wavelengths[0], fine_wavelengths[-1], num=fine_wavelengths.size)
    fine_transit_grid = griddata(fine_wavelengths, transit_depth, xi=fine_um_grid, method='linear')

    out_wavelengths, pixel_transit_depth = gen_measured_transit(R=R, pixel_bins=pixel_wavelengths,
                                                                fine_wl=fine_um_grid,
                                                                fine_transit=fine_transit_grid)
    '''end sample data into spectrum'''

    return out_wavelengths, pixel_transit_depth


def transit_spectra_no_h2o(pixel_wavelengths, theta, fixed, p0=1):
    # fixed global variables
    global mass_h2  # mean molecular weight of H2 He mix
    global mass_water
    global mass_co
    global mass_hcn

    # unpack model variables
    rad_planet, T, log_fco, log_fhcn = theta

    # unpack log ratios

    co_ratio = 10**log_fco
    hcn_ratio = 10**log_fhcn

    # package the ratios into a summable list
    trace_ratios = np.array([
        [co_ratio],
        [hcn_ratio]
    ])

    # print('trace ratios', trace_ratios)
    # unpack known priors
    fine_wavelengths, h2o_cross_sections, co_cross_sections, hcn_cross_sections, h2_cross_sections, g_planet, rad_star, R = fixed

    # package the cross sections into an array
    sigma_trace = np.array(
        [co_cross_sections,
         hcn_cross_sections]
    )


    # he/h2 ratio is disabled, since I don't have the cross-sections
    # h2he_ratio = .17
    # mass_h2he = (1 - h2he_ratio) * 2 + h2he_ratio * 4
    # mass = (1 - water_ratio) * mass_h2he + water_ratio * mass_water

    # temporary mass cause I don't have He cross sections yet
    sum_ratios = co_ratio + hcn_ratio
    mass = (1 - sum_ratios)*mass_h2 + co_ratio*mass_co + hcn_ratio*mass_hcn

    transit_depth = alpha_lambda(sigma_trace=sigma_trace,
                                 xi=trace_ratios,
                                 planet_radius=rad_planet,
                                 p0=p0,
                                 T=T,
                                 mass=mass,
                                 g=g_planet,
                                 star_radius=rad_star,
                                 sigma_filler=h2_cross_sections
                                 )

    '''Sample the data into a spectrum'''
    # set the resolution of the spectrometer
    # flip the data to ascending order
    fine_wavelengths = np.flip(fine_wavelengths)
    transit_depth = np.flip(transit_depth)

    # interpolate the data to even spacing
    fine_um_grid = np.linspace(fine_wavelengths[0], fine_wavelengths[-1], num=fine_wavelengths.size)
    fine_transit_grid = griddata(fine_wavelengths, transit_depth, xi=fine_um_grid, method='linear')

    out_wavelengths, pixel_transit_depth = gen_measured_transit(R=R, pixel_bins=pixel_wavelengths,
                                                                fine_wl=fine_um_grid,
                                                                fine_transit=fine_transit_grid)
    '''end sample data into spectrum'''

    return out_wavelengths, pixel_transit_depth


def transit_model_NULL(pixel_wavelengths, theta, fixed, p0=1):
    '''
    Model of the null case, defined as filler gas only.

    Parameters
    ----------
    pixel_wavelengths
    theta
    fixed
    p0

    Returns
    -------

    '''

    global mass_h2

    # unpack model variables
    rad_planet, T = theta

    # no species ratios, set to zero
    trace_ratios = 0

    # print('trace ratios', trace_ratios)
    # unpack known priors
    fine_wavelengths, h2_cross_sections, g_planet, rad_star, R = fixed

    # package the cross sections into an array
    sigma_trace = 0


    # he/h2 ratio is disabled, since I don't have the cross-sections
    # h2he_ratio = .17
    # mass_h2he = (1 - h2he_ratio) * 2 + h2he_ratio * 4
    # mass = (1 - water_ratio) * mass_h2he + water_ratio * mass_water

    # temporary mass cause I don't have He cross sections yet
    sum_ratios = np.sum(trace_ratios)
    weighted_mass_f = [mass_water, mass_ch4, mass_nh3, mass_hcn] * trace_ratios
    mass = (1 - sum_ratios)*mass_h2 + np.sum(weighted_mass_f)

    transit_depth = alpha_lambda(sigma_trace=sigma_trace,
                                 xi=trace_ratios,
                                 planet_radius=rad_planet,
                                 p0=p0,
                                 T=T,
                                 mass=mass,
                                 g=g_planet,
                                 star_radius=rad_star,
                                 sigma_filler=h2_cross_sections
                                 )

    '''Sample the data into a spectrum'''
    # set the resolution of the spectrometer
    # flip the data to ascending order
    fine_wavelengths = np.flip(fine_wavelengths)
    transit_depth = np.flip(transit_depth)

    # test prints
    # print('fine_wavelengths', fine_wavelengths.shape)
    # print('transit_depth', transit_depth.shape)

    out_wavelengths, pixel_transit_depth = gen_measured_transit(R=R, pixel_bins=pixel_wavelengths,
                                                                fine_wl=fine_wavelengths,
                                                                fine_transit=transit_depth)
    '''end sample data into spectrum'''

    return out_wavelengths, pixel_transit_depth


def transit_model_H2OCH4NH3HCN(pixel_wavelengths, theta, fixed, p0=1):
    # fixed global variables
    global mass_h2  # mean molecular weight of H2 He mix
    global mass_water
    global mass_ch4
    global mass_nh3
    global mass_hcn

    # unpack model variables
    rad_planet, T, log_h2o, log_ch4, log_nh3, log_hcn = theta

    # unpack log ratios
    # na_ratio = 10**log_na
    # k_ratio = 10**log_k
    water_ratio = 10**log_h2o
    ch4_ratio = 10**log_ch4
    nh3_ratio = 10**log_nh3
    hcn_ratio = 10**log_hcn

    # package the ratios into a summable list
    trace_ratios = np.array([
        [water_ratio],
        [ch4_ratio],
        [nh3_ratio],
        [hcn_ratio]
    ])

    # print('trace ratios', trace_ratios)
    # unpack known priors
    fine_wavelengths, water_cross_sections, ch4_cross_sections, nh3_cross_sections, hcn_cross_sections, h2_cross_sections, g_planet, rad_star, R = fixed

    # package the cross sections into an array
    sigma_trace = np.array(
        [water_cross_sections,
        ch4_cross_sections,
        nh3_cross_sections,
        hcn_cross_sections]
    )


    # he/h2 ratio is disabled, since I don't have the cross-sections
    # h2he_ratio = .17
    # mass_h2he = (1 - h2he_ratio) * 2 + h2he_ratio * 4
    # mass = (1 - water_ratio) * mass_h2he + water_ratio * mass_water

    # temporary mass cause I don't have He cross sections yet
    sum_ratios = np.sum(trace_ratios)
    weighted_mass_f = [mass_water, mass_ch4, mass_nh3, mass_hcn] * trace_ratios
    mass = (1 - sum_ratios)*mass_h2 + np.sum(weighted_mass_f)

    transit_depth = alpha_lambda(sigma_trace=sigma_trace,
                                 xi=trace_ratios,
                                 planet_radius=rad_planet,
                                 p0=p0,
                                 T=T,
                                 mass=mass,
                                 g=g_planet,
                                 star_radius=rad_star,
                                 sigma_filler=h2_cross_sections
                                 )

    '''Sample the data into a spectrum'''
    # set the resolution of the spectrometer
    # flip the data to ascending order
    fine_wavelengths = np.flip(fine_wavelengths)
    transit_depth = np.flip(transit_depth)

    # test prints
    # print('fine_wavelengths', fine_wavelengths.shape)
    # print('transit_depth', transit_depth.shape)

    out_wavelengths, pixel_transit_depth = gen_measured_transit(R=R, pixel_bins=pixel_wavelengths,
                                                                fine_wl=fine_wavelengths,
                                                                fine_transit=transit_depth)
    '''end sample data into spectrum'''

    return out_wavelengths, pixel_transit_depth


def transit_model_H2OCH4(pixel_wavelengths, theta, fixed, p0=1):
    # fixed global variables
    global mass_h2  # mean molecular weight of H2 He mix
    global mass_water
    global mass_ch4
    # global mass_nh3
    # global mass_hcn

    # unpack model variables
    rad_planet, T, log_h2o, log_ch4 = theta

    # unpack log ratios
    # na_ratio = 10**log_na
    # k_ratio = 10**log_k
    water_ratio = 10**log_h2o
    ch4_ratio = 10**log_ch4

    # package the ratios into a summable list
    trace_ratios = np.array([
        [water_ratio],
        [ch4_ratio]
    ])

    # print('trace ratios', trace_ratios)
    # unpack known priors
    fine_wavelengths, water_cross_sections, ch4_cross_sections, nh3_cross_sections, hcn_cross_sections, h2_cross_sections, g_planet, rad_star, R = fixed

    # package the cross sections into an array
    sigma_trace = np.array(
        [water_cross_sections,
         ch4_cross_sections]
    )


    # he/h2 ratio is disabled, since I don't have the cross-sections
    # h2he_ratio = .17
    # mass_h2he = (1 - h2he_ratio) * 2 + h2he_ratio * 4
    # mass = (1 - water_ratio) * mass_h2he + water_ratio * mass_water

    # temporary mass cause I don't have He cross sections yet
    sum_ratios = np.sum(trace_ratios)
    weighted_mass_f = [mass_water, mass_ch4] * trace_ratios
    mass = (1 - sum_ratios)*mass_h2 + np.sum(weighted_mass_f)

    transit_depth = alpha_lambda(sigma_trace=sigma_trace,
                                 xi=trace_ratios,
                                 planet_radius=rad_planet,
                                 p0=p0,
                                 T=T,
                                 mass=mass,
                                 g=g_planet,
                                 star_radius=rad_star,
                                 sigma_filler=h2_cross_sections
                                 )

    '''Sample the data into a spectrum'''
    # set the resolution of the spectrometer
    # flip the data to ascending order
    fine_wavelengths = np.flip(fine_wavelengths)
    transit_depth = np.flip(transit_depth)

    out_wavelengths, pixel_transit_depth = gen_measured_transit(R=R, pixel_bins=pixel_wavelengths,
                                                                fine_wl=fine_wavelengths,
                                                                fine_transit=transit_depth)
    '''end sample data into spectrum'''

    return out_wavelengths, pixel_transit_depth


def transit_model_H2O(pixel_wavelengths, theta, fixed, p0=1):
    return transit_spectra_test(pixel_wavelengths, theta, fixed, p0=p0)



def transit_model_CH4(pixel_wavelengths, theta, fixed, p0=1):
    # fixed global variables
    global mass_h2  # mean molecular weight of H2 He mix
    # global mass_water
    global mass_ch4

    # unpack model variables
    rad_planet, T, log_ch4 = theta

    # unpack log ratios
    # water_ratio = 10**log_h2o
    ch4_ratio = 10**log_ch4

    # package the ratios into a summable list
    trace_ratios = np.atleast_2d(ch4_ratio)

    # unpack known priors
    fine_wavelengths, h2o_cross_sections, h2_cross_sections, g_planet, rad_star, R = fixed

    # package the cross sections into an array
    sigma_trace = np.atleast_2d(h2o_cross_sections)

    sum_ratios = np.sum(trace_ratios)
    weighted_mass_f = [mass_ch4] * trace_ratios
    mass = (1 - sum_ratios)*mass_h2 + np.sum(weighted_mass_f)

    transit_depth = alpha_lambda(sigma_trace=sigma_trace,
                                 xi=trace_ratios,
                                 planet_radius=rad_planet,
                                 p0=p0,
                                 T=T,
                                 mass=mass,
                                 g=g_planet,
                                 star_radius=rad_star,
                                 sigma_filler=h2_cross_sections
                                 )

    '''Sample the data into a spectrum'''
    # set the resolution of the spectrometer
    # flip the data to ascending order
    fine_wavelengths = np.flip(fine_wavelengths)
    transit_depth = np.flip(transit_depth)

    out_wavelengths, pixel_transit_depth = gen_measured_transit(R=R, pixel_bins=pixel_wavelengths,
                                                                fine_wl=fine_wavelengths,
                                                                fine_transit=transit_depth)
    '''end sample data into spectrum'''

    return out_wavelengths, pixel_transit_depth
