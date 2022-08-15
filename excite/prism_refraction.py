import numpy as np

# refractive indices source:
# https://refractiveindex.info/?shelf=main&book=CaF2&page=Malitson
n_850 = 1.4300
n_3500 = 1.4140

# angle of minimum deviation
alpha_min = np.arcsin(n_850*np.sin(np.radians(30)))
print('angle of minimum deviation:', 2*np.degrees(alpha_min) - 60, 'degrees')


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


beta_min = output_angle(n_850, alpha=alpha_min)
print('output angle at minimum deviation:', np.degrees(beta_min), 'degrees')

beta_max = output_angle(n_3500, alpha=alpha_min)
dispersion_angle = beta_min - beta_max
print('total angle of dispersion:', np.degrees(dispersion_angle))

