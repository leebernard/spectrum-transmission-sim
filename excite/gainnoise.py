
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.modeling.physical_models import BlackBody as BB

from astropy.io import ascii

"""
# Model
1. Select a favourable target such as WASP-121b or similar
2. We will need the orbital period in seconds. So I convert period from days to seconds
3. Calculate a simple estimate of the eclipse depth, and use this as the peak-to-peak amplitude of the phase curve signal model.
4. The measured signal vs time is proportional to $F_s + F_p$, normalised to $F_s$, i.e. $1 + F_p/F_s$. In what follows, model is $F_p/F_s$
5. I also add a phase shift, that can be randomised in a further iteration of this work. 
"""

ColumnNames = ascii.read('./excite/tepcat-allplanets-csv.csv', data_start=0, data_end=1, )[0]
TepCat = ascii.read('./excite/tepcat-allplanets-csv.csv', data_start=1)
for column_name, TC_col in zip(ColumnNames, TepCat.colnames):
    if 'err' in column_name:
        TepCat.remove_column(TC_col)
    else:
        TepCat[TC_col].name = column_name
del ColumnNames
TepCat.add_index('System')

Name = 'WASP-121'
phase = 0.2*np.pi

'''See Gainnoise.ipynb'''

system = TepCat.loc[Name]
system['Period'] *= u.day.to(u.s)
PlanetSED = BB(temperature=system['Teq']*u.K, scale = 1*u.W/u.m**2/u.sr/u.micron)
StarSED   = BB(temperature=system['Teff']*u.K, scale = 1*u.W/u.m**2/u.sr/u.micron)

eclipse_depth = (system['R_b']*u.Rjup.to(u.m)/(system['R_A']*u.Rsun.to(u.m)))**2 * \
            PlanetSED(2.2*u.micron)/StarSED(2.2*u.micron)
eclipse_depth = eclipse_depth.value

delta_t = 60.0 # seconds->days
tt = np.arange(0, system['Period'], delta_t)

model  = 0.5*eclipse_depth*np.sin(2*np.pi*tt/system['Period'] + phase)

fig, ax = plt.subplots(1,1,figsize=(10, 5))
ax.plot(tt/3600, model)
ax.set_ylabel('Model')
ax.set_xlabel('Time [h]')
ax.grid()

'''See Gainnoise.ipynb'''

Nsim = 1024     # this many simulations
noise_std =1e-3 # this value to be replaced with ExoRad's estimate

# Repeat Nsim independent observations adding noise to each
# Observations are stored in D1's first dimension
D1  = np.zeros( (Nsim, tt.size) ) + model
D1 += np.random.randn( *D1.shape )*noise_std

# Least-square fit the data
A = np.c_[np.sin(2*np.pi*tt/system['Period']),
          np.cos(2*np.pi*tt/system['Period']),
          np.ones(tt.size)]

par, *_ = np.linalg.lstsq(A, D1.T, rcond=-1)

# Calculate statistics
amplitude = np.sqrt( par[0]**2 + par[1]**2)
amplitude_mean = amplitude.mean()
amplitude_std  = amplitude.std()
amplitude_snr  = amplitude_mean/amplitude_std
amplitude_bias = (0.5*eclipse_depth - amplitude_mean)/(0.5*eclipse_depth)
amplitude_bias_error = amplitude_std/(0.5*eclipse_depth)/np.sqrt(amplitude.size)
TITLE = 'SNR = {:.0f};  BIAS = ({:.2f} +/- {:.2f})%'.format(amplitude_snr,
                                           100*amplitude_bias,
                                           100*amplitude_bias_error)

fig, ax = plt.subplots(1,1,figsize=(10, 5))
ax.hist(100*amplitude)
ax.set_xlabel('Amplitude [%]')
ax.set_title(TITLE)
ax.vlines(0.5*eclipse_depth*100, *ax.get_ylim(), color='r')
ax.grid()

'''See Gainnoise.ipynb'''

# M1 temperature
TM1 = np.fromfile('./excite/t_prime_sf', dtype=np.uint16)
TM1 = 2.840909090909e-02*TM1 -2.731500000000e+02
# M2 temperature
TM2 = np.fromfile('./excite/t_second_sf', dtype=np.uint16)
TM2 = 2.840909090909e-02*TM2 -2.731500000000e+02

TM = 0.5*(TM1+TM2)
blast_time = np.arange(TM.size)/5.0

# Cut away firs two days worth of data as it contains glitches
TM = TM[864000:]
blast_time = blast_time[864000:]

# regrid blast data on the simulation time grid by
#  1. box car avaraging with kernel delta_t long
#  2. decimating at a delta_t cadence
#  note: now data have delta_t cadence
kernel = np.ones( np.int(delta_t*5) )
TM = np.convolve(TM, kernel/kernel.size, mode='same')[kernel.size::kernel.size]

fig, ax = plt.subplots(1,1,figsize=(10, 5))
ax.plot(np.arange(TM.size)*60/3600/25, TM)
ax.set_ylabel ( 'Temperature [$^o$C]')
ax.set_xlabel('Time [days]')
ax.grid()
_=_


'''See Gainnoise.ipynb'''

Nsim = 1024     # this many simulations
noise_std =1e-3 # this value to be replaced with ExoRad's estimate
trend_amplitude = 0.001

# Repeat Nsim independent observations adding noise to each
# Observations are stored in D1's first dimension
D1  = np.zeros( (Nsim, tt.size) ) + model
D1 += np.random.randn( *D1.shape )*noise_std
TData = np.zeros_like(D1)

for k in range(D1.shape[0]):
    idx0 = np.random.randint(0, high=TM.size  - D1.shape[1])
    temp_trend = TM[idx0:idx0+D1.shape[1]].copy()
    TData[k] = temp_trend.copy()
    temp_trend -= temp_trend.mean()
    temp_trend /= np.abs(temp_trend).max()
    temp_trend = trend_amplitude*temp_trend
    D1[k] += temp_trend

fig, ax = plt.subplots(1,1,figsize=(10, 5))
ax.set_ylabel ( 'Signal')
ax.set_xlabel('Time [h]')
ax.grid()

for k in range(100):
    ax.plot(tt/3600, D1[k])

'''Do statistics '''
amplitude = np.zeros(D1.shape[0])

for k in range(D1.shape[0]):

    A = np.c_[np.sin(2*np.pi*tt/system['Period']),
              np.cos(2*np.pi*tt/system['Period']),
              TData[k],
              np.ones(tt.size)]

    par, *_ = np.linalg.lstsq(A, D1[k], rcond=-1)
    amplitude[k] = np.sqrt( par[0]**2 + par[1]**2)

# Calculate statistics
amplitude_mean = amplitude.mean()
amplitude_std  = amplitude.std()
amplitude_snr  = amplitude_mean/amplitude_std
amplitude_bias = (0.5*eclipse_depth - amplitude_mean)/(0.5*eclipse_depth)
amplitude_bias_error = amplitude_std/(0.5*eclipse_depth)/np.sqrt(amplitude.size)
TITLE = 'SNR = {:.0f};  BIAS = ({:.2f} +/- {:.2f})%'.format(amplitude_snr,
                                           100*amplitude_bias,
                                           100*amplitude_bias_error)

fig, ax = plt.subplots(1,1,figsize=(10, 5))
ax.hist(100*amplitude)
ax.set_xlabel('Amplitude [%]')
ax.set_title(TITLE)
ax.vlines(0.5*eclipse_depth*100, *ax.get_ylim(), color='r')
ax.grid()

'''My analysis starts here'''
# make a fourier transform of tdata
F_TData = []

for k in range(TData.shape[0]):
    F_TData.append(np.fft.fft(TData[k]))

F_TData = np.array(F_TData)

# make a plot of it
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for k in range(100):
    ax.plot(np.abs(F_TData[k]))
