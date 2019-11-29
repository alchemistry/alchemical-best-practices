from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

colormap = cm.cividis # Not sure the best one to use.  We want more distincttion lower.  
fig = plt.figure()
ax = fig.gca(projection='3d')

rmin = 0.1
rmax = 9.0
ngrid = 100
alpha = 0.5
sigma = 3.0
eps = 1 
zmax = 8

r = np.linspace(rmin, rmax, ngrid)
lam = np.linspace(0,1,ngrid)
r,lam = np.meshgrid(r,lam)
lamr = 1.0/(alpha*(1.0-lam) + (r/sigma)**6)
u = 4*eps*lam*(lamr**2 - lamr)
u[u>zmax] = zmax

surf = ax.plot_surface(r, lam, u, cmap=cm.cividis,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-eps, zmax)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_title('The softcore interaction potential')
ax.set_xlabel('Particle distance')
ax.set_ylabel(r'$\lambda$')
ax.set_zlabel('Energy')

# add lines at lam = 0 and lam = 1 to show the physical endpoints
rrange = np.linspace(rmin,rmax,ngrid) 
ax.plot(rrange,np.zeros(ngrid),np.zeros(ngrid),color='r',linewidth=3)
lj = 4*eps*((rrange/sigma)**(-12) - (rrange/sigma)**(-6))
lj[lj>zmax] = np.nan
ax.plot(rrange,np.ones(ngrid),lj,color='r',linewidth=3)

ax.view_init(elev=20., azim=-65)
plt.savefig("lj_softcore.pdf")

#plt.show()
