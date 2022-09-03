#
# How do you get dtheta/dlambda?  For a prism, look up n(lambda),
import numpy as np

# wavelength grid, in microns
l = np.linspace(0.8,3.5,1024); l2=l*l

# for CaF2 prism:
a0, a1, a2 = 0.69913, 0.11994, 4.35181
b0, b1, b2 = 0.093742**2, 21.182**2, 38.46**2
n = np.sqrt( 1.33973 + a0*l2/(l2-b0) + a1*l2/(l2-b1) + a2*l2/(l2-b2) )
dn_dl = 1./n*abs( a0*b0*l/(l2-b0)**2 + a1*b1*l/(l2-b1)**2 + a2*b2*l/(l2-b2)**2 )  # min of ~4.67e-3 at l~1.55 microns

# prism geometry, snell's law: n*sin(0.5*alpha) = sin(0.5*(alpha+theta)) is symmetric passage, can be chosen for one wavelength
#  alpha=60 is wedge angle and theta is deflection angle
i0=abs(l-1.).argmin()
n0=n[i0]

# trace rays trough prism, theta0 -> theta0p (first surface) and then theta1 -> theta1p (second surface)
#  see, e.g., https://en.wikipedia.org/wiki/Prism
# alpha = 60*pi/180.
# theta0 = arcsin(n0*sin(alpha/2))
# theta0p = arcsin(sin(theta0)/n)   # 1st surface: theta_i is theta0 (index 1), theta_t is theta0p (index n)
# theta1 = alpha - theta0p          # relation between internal angles
# theta1p = arcsin(sin(theta1)*n)   # 2nd surface: theta_i is theta1 (index n), theta_t is theta1p (index 1)
# theta = theta0 + theta1p - alpha  # total deflection angle
#
#   but we just need dtheta_dn:
# from chain rule: dtheta_dn = 1/cos(theta1p) * ( sin(theta1) + cos(theta1)*1./cos(theta0p) * n0/n * sin(alpha/2) )
#                            = sin(alpha)/cos(theta0p)/cos(theta1p), after some algebra and trig substitutions
s1p = 0.5*np.sqrt(3)*np.sqrt(n**2-(0.5*n0)**2)-0.25*n0  	#  sin(theta1p) ~ n0*sin(alpha/2), or theta1p ~ theta0
s0p = 0.5*n0/n                                    	#  sin(theta0p) ~ sin(alpha/2), or theta0p ~ alpha/2
dtheta_dn = 0.5*np.sqrt(3)/np.sqrt(1-s0p**2)/np.sqrt(1-s1p**2)   # ~2*sin(alpha/2)/cos(theta0) ~ n
dtheta_dl = dtheta_dn*dn_dl

#
# current design has F1=4 inch collimator, and F2=10.7 inch camera mirror
#   telescope has F-number, Fn=12
F1,F2,Fn=4*2.54*1.e-2,272.24*1.e-3,12.

# diffraction limited, using formula delta_theta = lambda/D (or delta_theta = sw/(D*Fn))
#   R = F2*l*dtheta_dl/(delta_theta*Feff) = F2*dtheta_dl/F#eff = F1*dtheta_dl/Fn
R=dtheta_dl*F1/Fn*1.e6

# you can also see how the spread over pixels works
ps=18.e-6  # H2RG pixel size
x = F2*dtheta_dl.cumsum()*(l[1]-l[0]) / ps

# reflection losses? 
alpha = 60*np.pi/180.
theta0 = np.arcsin(n[i0]*np.sin(alpha/2))
theta0p = np.arcsin(np.sin(theta0)/n) # 1st surface: theta_i is theta0 (index 1), theta_t is theta0p (index n)
theta1 = alpha - theta0p
theta1p = np.arcsin(np.sin(theta1)*n) # 2nd surface: theta_i is theta1 (index n), theta_t is theta1p (index 1)
theta = theta0 + theta1p - alpha

# assume that the collimating mirror is aligned with theta[i0] coming in normal
#  that is symmetric passage, for which the incoming and outgoing beams are the same size
# at other wavelengths, the beam size will be:

beam_size = np.cos(theta-theta[i0])
Fnum_eff_dispersion = Fn*F2/F1/beam_size
print ('anamorphic ratio', Fnum_eff_dispersion.max()/Fnum_eff_dispersion.min())
# it's a very small change since we're working very close to symmetric passage

# reflection losses (Fresnel equations), T is total transmission through prism
Rs1 = ( -np.sin(theta0-theta0p)/np.sin(theta0+theta0p) )**2
Rp1 = ( np.tan(theta0-theta0p)/np.tan(theta0+theta0p) )**2
Rs2 = ( -np.sin(theta1p-theta1)/np.sin(theta1+theta1p) )**2
Rp2 = ( np.tan(theta1p-theta1)/np.tan(theta1p+theta1) )**2
T = (1-0.5*(Rs1+Rp1))*(1-0.5*(Rs2+Rp2))
