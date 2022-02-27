from spot0 import spot
import matplotlib.pyplot as plt

ps, F1, F2, sep = 18.e-6, 4*2.54*1.e-2, 10.7*2.54*1.e-2, 6*2.54*1.e-2

x, y, z, dx, dy, dz = spot(theta_1=90., theta_2=30., sep=sep, efl1=F1, efl2=F2)


plt.plot( (y[1:] - y[0])/ps, (z[1:]-z[0])/ps, 'o')


