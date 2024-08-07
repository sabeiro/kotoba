import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpmath import *

# Real part of spherical harmonic Y_(4,0)(theta,phi)
def Y(l,m):
    def g(theta,phi):
        R = abs(fp.re(fp.spherharm(l,m,theta,phi)))
        x = R*fp.cos(phi)*fp.sin(theta)
        y = R*fp.sin(phi)*fp.sin(theta)
        z = R*fp.cos(theta)
        return [x,y,z]
#        return [x,z]
    return g

#fp.splot(Y(4,0), [0,fp.pi], [0,2*fp.pi], points=100)
fp.cplot(fp.gamma, points=100000)
#cplot(lambda z: z, [-2, 2], [-10, 10])
#cplot(exp)
#cplot(zeta, [0, 1], [0, 50])


#fp.cplot(Y(4,0), [0,fp.pi],  [0,2*fp.pi], points=100)
# fp.splot(Y(4,0), [0,fp.pi], [0,2*fp.pi], points=300)
# fp.splot(Y(4,1), [0,fp.pi], [0,2*fp.pi], points=300)
# fp.splot(Y(4,2), [0,fp.pi], [0,2*fp.pi], points=300)
# fp.splot(Y(4,3), [0,fp.pi], [0,2*fp.pi], points=300)
