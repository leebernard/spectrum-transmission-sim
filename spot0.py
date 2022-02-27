from numpy import sqrt,cos,sin,pi,arange,newaxis,hstack
# from rotsky import rotsky

def spot(x00=0., y00=0., z00=0., x11=0., phi=0, theta_1=90., theta_2=30., dtheta=0., Fn=12., efl1=4.,efl2=10.7, sep=6., N=203, recenter=False, rotate_detector=False, step=3):
    """
      sending focused light through an 2 OAP relay
        (default units are degrees and inches)

      x00,y00,z00: starting focal plane positions
      phi : rotation about optical axis for deflection in degrees
      theta_1 : deflection angle of first OAP (collimator) in degrees
      theta_2 : deflection angle of second OAP (camera) in degrees
      dtheta: shift about z-axis in collimated beam (e.g., due to prism)
      Fn : telescope F-number
      ef1,ef2: (effective) focal lengths of two mirrors
      N : number of rays to trace (overrides to len(theta) if vector)

      returns: x,y positions (same units as efl1,efl2) of the traced rays on the detector
    """

    #
    # fill the telescope aperture, take rays to x00,y00,z00
    Rs, Rt = 0.095, 0.25
    x = (arange(N)-N//2) * Rt/(N//2)
    xx = x[:, newaxis] + 0*x[newaxis, :]
    yy = 0*x[:, newaxis] + x[newaxis, :]
    rad2 = xx**2 + yy**2
    j = (rad2 > Rs**2)*(rad2 <= Rt**2)
    z = xx[j]; y = yy[j]; x = 2*Rt*Fn+0*y
    xx=0;yy=0;rad2=0;j=0

    print ("""Beam Size 0: %.4f, %.4f (%.4f)""" % (z.max()-z.min(), y.max()-y.min(), 2*Rt))

    dx, dy, dz = x-x00, y-y00, z-z00
    # add in chief ray
    dx = hstack((2*Rt*Fn-x00, dx))
    dy = hstack((-y00, dy))
    dz = hstack((-z00, dz))
    norm = sqrt(dx**2+dy**2+dz**2)
    dx/=norm; dy/=norm; dz/=norm

    # miror focal lengths parameters
    th1, th2 = theta_1*pi/180, theta_2*pi/180.
    f1 = 0.5*efl1*(1+cos(th1))
    f2 = 0.5*efl2*(1+cos(th2))

    # rotate to oap coordinate system
    dx, dz = dx*sin(th1) + dz*cos(th1), dz*sin(th1) - dx*cos(th1)
    x0, z0, y0 = x00*sin(th1) + z00*cos(th1), z00*sin(th1) - x00*cos(th1), 1.*y00

    # x,y,z = ef*sin(th),0.,0.5*ef*(1-cos(th))
    # x0 + s*dx[0] = efl1*sin(th1)
    # y0 + s*dy[0] = 0
    # z0 + s*dz[0] = -efl1*cos(th1)
    if (recenter):
        s0 = (efl1*sin(th1)-x0)/dx[0]
        y0 -= y0+s0*dy[0]
        z0 -= z0+s0*dz[0] + efl1*cos(th1)

    # path step to oap surface at z = (x^2+y^2)/(4*f1), with z=z0+f1+s*dz
    s = ( 2*f1*dz-x0*dx-y0*dy + sqrt( (2*f1*dz-x0*dx-y0*dy)**2 + (dx**2+dy**2)*(4*f1*(f1+z0)-x0**2-y0**2) ) )/(dx**2+dy**2)

    x, y ,z = x0+s*dx, y0+s*dy, z0+f1+s*dz
    if (step==0): return x, y, z, dx, dy, dz

    print ("""Beam Size 1: %.4f, %.4f (%.4f)""" % (x.max()-x.min(), y.max()-y.min(), efl1/Fn))

    # bounce off the first mirror, normal vector n
    nx, ny, nz = -x/(2*f1), -y/(2*f1), 1.
    norm = sqrt(nx*nx+ny*ny+nz*nz)
    nx/=norm; ny/=norm; nz/=norm
    vdotn = dx*nx+dy*ny+dz*nz
    # reflection from vector d to r: r = d-2*(d.n)*n
    dx1, dy1, dz1 = dx-2*vdotn*nx, dy-2*vdotn*ny, dz-2*vdotn*nz

    # move up
    s = 0.5*sep/dz1
    x, y, z = x+s*dx1, y+s*dy1, z+s*dz1

    # reflect back
    dz1 = -dz1

    # prism
    dth = dtheta*pi/180.
    dx1, dz1 = dx1*cos(dth)+dz1*sin(dth), dz1*cos(dth)-dx1*sin(dth)

    # center spot on 2nd mirror
    x = efl1*sin(th1) - x - efl2*sin(th2)
    #x = (x-efl1*sin(th1)) + efl2*sin(th2)

    if (recenter):
        #z[0] + s0*dz1[0] = 0.5*efl2*(1-cos(th2))
        s0 = ( 0.5*efl2*(1-cos(th2))-z[0] )/dz1[0]
        print ("s0:", s0)
        y -= y[0]+s0*dy1[0]
        #x -= x[0]+s0*dx1[0] - efl2*sin(th2)
        x -= x[0]+s0*dx1[0] + efl2*sin(th2)

    # path length to 2nd mirror
    # s^2*(dx1^2+dy1^2) + 2*s*(x*dx1+y*dy1-2*f2*dz1) + (x^2+y^2-4*f2*z) = 0
    # s = 2*c/( -b -/+ sqrt(b^2-4*a*c) )
    s = (x**2+y**2-4*f2*z)/( 2*f2*dz1-x*dx1-y*dy1 - sqrt( (2*f2*dz1-x*dx1-y*dy1)**2 + (dx1**2+dy1**2)*(4*f2*z-x**2-y**2) ) )
    x += s*dx1; y += s*dy1; z += s*dz1;
   
    if (step==1): return x, y, z, dx1, dy1, dz1

    print ("""Beam Size 2: %.4f, %.4f (%.4f)""" % (x.max()-x.min(),y.max()-y.min(),efl1/Fn))

    # focus the spot with mirror 2
    #z = (x**2+y**2)/(4*f2)
    nx, ny, nz = -x/(2*f2), -y/(2*f2), 1.
    norm = sqrt(nx*nx+ny*ny+nz*nz)
    nx /= norm; ny /= norm; nz /= norm
    vdotn = dx1*nx+dy1*ny+dz1*nz
    dx2 = dx1-2*vdotn*nx
    dy2 = dy1-2*vdotn*ny
    dz2 = dz1-2*vdotn*nz

    # rotate from oap coordinates
    #x,z = x*sin(th2) - (z-f2)*cos(th2) , (z-f2)*sin(th2) + x*cos(th2)
    #dx2,dz2 = dx2*sin(th2) - dz2*cos(th2) , dz2*sin(th2) + dx2*cos(th2)
    x, z = x*sin(th2) + (z-f2)*cos(th2), (z-f2)*sin(th2) - x*cos(th2)
    dx2, dz2 = dx2*sin(th2) + dz2*cos(th2), dz2*sin(th2) - dx2*cos(th2)

    if step==2:
        return x, y, z, dx2, dy2, dz2

    # assume detector is at x=0, rotated by theta2/2
    # x*cos(th2/2) = z*sin(th2/2)
    if rotate_detector:
        #s = -( (x+x11)*cos(th2/2) - z*sin(th2/2) )/( dx2*cos(th2/2) - dz2*sin(th2/2) )
        s = -( (x+x11)*cos(th2/2) + z*sin(th2/2) )/( dx2*cos(th2/2) + dz2*sin(th2/2) )
    else:
        s = -(x+x11)/dx2

    x2, y2, z2 = x+s*dx2, y+s*dy2, z+s*dz2

    if step==3:
        return x2, y2, z2, dx2, dy2, dz2
