import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator

from spot0 import spot

figsize = (6, 6)

ps = 18.e-6  # pixel pitch (meters)
F1 = 4*2.54*1.e-2  # focal length of 1st OAP (meters)
F2 = 10.7*2.54*1.e-2  # focal length of 2nd OAP (meters)
sep = 6*2.54*1.e-2  # this is converting inches to meters

x, y, z, dx, dy, dz = spot(theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)

fig, ax = plt.subplots(figsize=figsize)
ax.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o')
ax.set_xlim(1, -1)
ax.set_ylim(1, -1)
ax.set_title('Ideally aligned spectrograph')


'''Focus misalignment'''
oof = 100
focus_offset = oof*1e-6  # convert microns to meters
fig2, ax2 = plt.subplots(figsize=figsize)
alpha = 0.3
ax2.set_title('Out of focus (oof) examples')
ax2.set_xlabel('Pixels')
ax2.grid(True)
# all units are in meters
slit_x = 0.00  # offset in the optical axis direction (focus)
slit_y = 0.00  # offset parallel to slit
slit_z = 0.0  # offset perpendicular to slit
detector_offset = 0.00 # offset in detector position (equiv to 2nd OAP focus)
x, y, z, dx, dy, dz = spot(x00=slit_x, y00=slit_y, z00=slit_z, x11=detector_offset,  recenter=True,
                           theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)

ax2.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o', alpha=alpha, markeredgewidth=0.0, label='Ideal alignment')

# all units are in meters
slit_x = focus_offset  # offset in the optical axis direction (focus)
slit_y = 0.00  # offset parallel to slit
slit_z = 0.0  # offset perpendicular to slit
detector_offset = 0.00 # offset in detector position (equiv to 2nd OAP focus)
x, y, z, dx, dy, dz = spot(x00=slit_x, y00=slit_y, z00=slit_z, x11=detector_offset,  recenter=True,
                           theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)
ax2.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o', alpha=alpha, markeredgewidth=0.0, label='OAP 1 oof by %dμm' % oof)

# all units are in meters
slit_x = 0.00  # offset in the optical axis direction (focus)
slit_y = 0.00  # offset parallel to slit
slit_z = 0.0  # offset perpendicular to slit
detector_offset = focus_offset # offset in detector position (equiv to 2nd OAP focus)
x, y, z, dx, dy, dz = spot(x00=slit_x, y00=slit_y, z00=slit_z, x11=detector_offset,  recenter=True,
                           theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)
ax2.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o', alpha=alpha, markeredgewidth=0.0, label='OAP 2 oof by %dμm' % oof)

slit_x = focus_offset  # offset in the optical axis direction (focus)
slit_y = 0.00  # offset parallel to slit
slit_z = 0.0  # offset perpendicular to slit
detector_offset = focus_offset # offset in detector position (equiv to 2nd OAP focus)
x, y, z, dx, dy, dz = spot(x00=slit_x, y00=slit_y, z00=slit_z, x11=detector_offset,  recenter=True,
                           theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)
ax2.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o', alpha=alpha, markeredgewidth=0.0, label='Both OAP 1 and 2 %dμm' % oof)

squ_apt = 2
ax2.set_xlim(squ_apt/2, -squ_apt/2)
ax2.set_ylim(squ_apt/2, -squ_apt/2)
plt.legend()


'''Angle misalignment'''
# set angle out of alignment, using small angle approx
# 1 arcmin to start
arcmin = 60  # 1/2 degree
offset = arcmin/60*np.pi/180 * F1 # convert armins to effective slit offset, using SSA

fig2, ax2 = plt.subplots(figsize=figsize)
alpha = 0.3
ax2.set_title('OAP misalignment examples')
ax2.set_xlabel('Pixels')
ax2.grid(True)
# all units are in meters

slit_x = 0.00  # offset in the optical axis direction (focus)
slit_y = 0.00 # offset parallel to slit
slit_z = 0.00  # offset perpendicular to slit
detector_offset = 0.00 # offset in detector position (equiv to 2nd OAP focus)
x, y, z, dx, dy, dz = spot(x00=slit_x, y00=slit_y, z00=slit_z, x11=detector_offset,  recenter=True,
                           theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)

ax2.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o', alpha=alpha, markeredgewidth=0.0, label='Ideal alignment')

slit_x = 0.00  # offset in the optical axis direction (focus)
slit_y = offset # offset parallel to slit
slit_z = 0.00  # offset perpendicular to slit
detector_offset = 0.00 # offset in detector position (equiv to 2nd OAP focus)
x, y, z, dx, dy, dz = spot(x00=slit_x, y00=slit_y, z00=slit_z, x11=detector_offset,  recenter=True,
                           theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)

ax2.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o', alpha=alpha, markeredgewidth=0.0, label='1st OAP misaligned vertically by %d arcmins' % arcmin)

slit_x = 0.00  # offset in the optical axis direction (focus)
slit_y = 0.00 # offset parallel to slit
slit_z = offset  # offset perpendicular to slit
detector_offset = 0.00 # offset in detector position (equiv to 2nd OAP focus)
x, y, z, dx, dy, dz = spot(x00=slit_x, y00=slit_y, z00=slit_z, x11=detector_offset,  recenter=True,
                           theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)

ax2.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o', alpha=alpha, markeredgewidth=0.0, label='1st OAP misaligned horizontially by %d arcmins' % arcmin)


oof = 0
focus_offset = oof*1e-6  # convert microns to meters
slit_x = 0.00  # offset in the optical axis direction (focus)
slit_y = offset/np.sqrt(2) # offset parallel to slit
slit_z = offset/np.sqrt(2)  # offset perpendicular to slit
detector_offset = focus_offset # offset in detector position (equiv to 2nd OAP focus)
x, y, z, dx, dy, dz = spot(x00=slit_x, y00=slit_y, z00=slit_z, x11=detector_offset,  recenter=True,
                           theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)

ax2.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o', alpha=alpha, markeredgewidth=0.0, label='1st OAP misaligned diagionally by %d arcmins' % arcmin)

squ_apt = 1
ax2.set_xlim(squ_apt/2, -squ_apt/2)
ax2.set_ylim(squ_apt/2, -squ_apt/2)
ax2.legend()


'''Combination of effects'''
fig3, ax3 = plt.subplots(figsize=figsize)
alpha = 0.3
ax3.set_title('OAP misalignment examples')
ax3.set_xlabel('Pixels')
ax3.grid(True)

slit_x = 0.00  # offset in the optical axis direction (focus)
slit_y = offset/np.sqrt(2) # offset parallel to slit
slit_z = offset/np.sqrt(2)  # offset perpendicular to slit
detector_offset = focus_offset # offset in detector position (equiv to 2nd OAP focus)
x, y, z, dx, dy, dz = spot(x00=slit_x, y00=slit_y, z00=slit_z, x11=detector_offset,  recenter=True,
                           theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)

ax3.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o', alpha=alpha, markeredgewidth=0.0, label='1st OAP misaligned diagionally by %d arcmins' % arcmin)

oof = 100
focus_offset = oof*1e-6  # convert microns to meters
# all units are in meters
slit_x = focus_offset  # offset in the optical axis direction (focus)
slit_y = 0.00  # offset parallel to slit
slit_z = 0.0  # offset perpendicular to slit
detector_offset = 0.00 # offset in detector position (equiv to 2nd OAP focus)
x, y, z, dx, dy, dz = spot(x00=slit_x, y00=slit_y, z00=slit_z, x11=detector_offset,  recenter=True,
                           theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)
ax3.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o', alpha=alpha, markeredgewidth=0.0, label='OAP 1 oof by %dμm' % oof)

slit_x = 0.00  # offset in the optical axis direction (focus)
slit_y = offset/np.sqrt(2) # offset parallel to slit
slit_z = offset/np.sqrt(2)  # offset perpendicular to slit
detector_offset = focus_offset # offset in detector position (equiv to 2nd OAP focus)
x, y, z, dx, dy, dz = spot(x00=slit_x, y00=slit_y, z00=slit_z, x11=detector_offset,  recenter=True,
                           theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)

ax3.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o', alpha=alpha, markeredgewidth=0.0, label='1st OAP misaligned diagionally by %d arcmins and misfocused by %d' % (arcmin, oof))

squ_apt = 0.5
ax3.set_xlim(squ_apt/2, -squ_apt/2)
ax3.set_ylim(squ_apt/2, -squ_apt/2)
ax3.legend()
