'''
calculations for the Thorlabs KT311 - Spatial Filter System
The rails are  2 inches (50 mm) long, ie 50 mm max focal distance
'''

import numpy as np


def filter_diam(f1, beam_dia, wavelength):
    d = 2*f1*wavelength/beam_dia
    print('filter diameter:', d, 'um')
    return d


def magnification(f1, f2, beam_dia):
    output_dia = f2/f1 * beam_dia
    print('output beam:', output_dia)
    return output_dia

d_beam = 5.0  # mm
wavelength = 0.655  # um

f_15 = 15.29  # mm
focus_8 = 8.0  # mm
f_96 = 9.6
f_4 = 4.51  # mm
col_30 = 30.0  # mm


d_hene = 0.63  # mm
wavelength_hene = 0.6328  # um

f_276 = 2.76
f_3 = 3.1
f_2 = 2.0
f_55 = 5.5
col_40 = 40.0
col_45 = 45.0
col_50 = 50.0

print('Circularization of a 655nm laser with long diameter 5mm')
filter_1, beam_1 = filter_diam(focus_8, d_beam, wavelength), magnification(focus_8, col_30, d_beam)

print('Circularization of a 655nm laser with long diameter 5mm')
filter_2, beam_2 = filter_diam(focus_8, d_beam, wavelength), magnification(focus_8, col_40, d_beam)



print('Circularization of an HeNe laser with beam diameter 0.63mm')
filter_hene1, beam_hene1 = filter_diam(2.75, d_hene, wavelength_hene), magnification(2.75, col_50, d_hene)



