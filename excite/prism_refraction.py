import numpy as np


def sellmeier(wl, B1=5.676e-1, C1=2.526e-3, B2=4.711e-1, C2=1.008e-2, B3=3.848, C3=1.201e3):
    '''
    Sellmeier equation. Returns index of refraction squared (n**2).
    Default is for calcium floride (CaF2)
    Source: https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=148&pn=PS863

    Parameters
    ----------
    wavelength: in um
    B1
    B2
    B3
    C1
    C2
    C3

    Returns
    -------

    '''
    return 1 + B1*wl**2/(wl**2 - C1) + B2*wl**2/(wl**2 - C2) + B3*wl**2/(wl**2 - C3)


def output_angle(n, alpha=np.radians(30)):
    '''
    Output angle of a prism, as a function of index of refraction,
    parameterized by input angle. Assumes an equilateral prism.

    Parameters
    ----------
    n: index of refraction
    alpha: angle in radians

    Returns
    -------
    Output angle relative to the prism face, in radians
    '''

    return np.arcsin(n * np.sin(np.radians(60)) - np.arcsin(1/n * np.sin(alpha)))


# refractive indices source:
# https://refractiveindex.info/?shelf=main&book=CaF2&page=Malitson
n_850 = np.sqrt(sellmeier(0.850))
n_1000 = np.sqrt(sellmeier(1.00))
n_3500 =np.sqrt(sellmeier(3.50))

# angle of minimum deviation
alpha_min = np.arcsin(n_850*np.sin(np.radians(30)))
print('angle of minimum deviation, 0.85 um:', 2*np.degrees(alpha_min) - 60, 'degrees')

beta_min = output_angle(n_850, alpha=alpha_min)
print('output angle at minimum deviation, 0.85 um:', np.degrees(beta_min), 'degrees')

beta_max = output_angle(n_3500, alpha=alpha_min)
dispersion_angle = beta_min - beta_max
print('total angle of dispersion for 0.85-3.50 um:', np.degrees(dispersion_angle))


# angle of minimum deviation again, for 1.00 um
alpha_min = np.arcsin(n_1000*np.sin(np.radians(30)))
print('angle of minimum deviation, 1.00 um:', 2*np.degrees(alpha_min) - 60, 'degrees')

beta_min = output_angle(n_1000, alpha=alpha_min)
print('output angle at minimum deviation, 1.00 um:', np.degrees(beta_min), 'degrees')

beta_max = output_angle(n_3500, alpha=alpha_min)
dispersion_angle = beta_min - beta_max
print('total angle of dispersion for 1.00-3.50 um:', np.degrees(dispersion_angle))



