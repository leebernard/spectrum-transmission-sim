import numpy as np
from toolkit import spectrum_slicer

# filename = '/home/lee/natlab/excite_optic_data/CaF2_Transmission.xlsx'
filename = '/home/lee/natlab/excite_optic_data/caf2_trans.dat'

caf2_rawdata = np.loadtxt(filename, skiprows=2)

resolution = 10

start_wl = 0.400
end_wl = 4.00

# convert to um from nm
caf2_rawdata[:, 0] *= 1/1000

# slice data
temp = spectrum_slicer(start_wl, end_wl, dataset=caf2_rawdata)



