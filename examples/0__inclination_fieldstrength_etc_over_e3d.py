from datetime import datetime
import numpy as np

from matplotlib import pyplot as plt
plt.ion()

import ppigrf

import e3doubt
from importlib import reload
reload(e3doubt)
from e3doubt import experiment

from e3doubt.utils import get_supported_sites
from e3doubt.geodesy import geodetic2geocentricXYZ,ECEF2geodetic,geod2geoc

import e3doubt.radar_utils
reload(e3doubt.radar_utils)
from e3doubt.radar_utils import get_enu_vectors_cartesian,get_receiver_az_el_geod,ECEFtosphericalENUMatrix

sites = get_supported_sites()

igrf_refdate = datetime(2020,12,1)

gdlat_t, glon_t = sites.loc['SKI']
xt, yt, zt = geodetic2geocentricXYZ(gdlat_t, glon_t, 0, returnR = False, degrees = True)

##############################
# Find az, el where Skibotn transceiver aligns best with stuff

# 1. Generate a bunch of points over Skibotn
h = 300                         # km
ddeg = 0.01
degdim = 3.0
dlat = np.arange(-degdim,degdim,ddeg)
dlon = np.arange(-degdim,degdim,ddeg)

dlat, dlon = np.meshgrid(dlat,dlon,indexing='ij')
dlshape = dlat.shape

dlat, dlon = dlat.ravel(), dlon.ravel()

gdlatp = gdlat_t + dlat
glonp = glon_t+dlon

xp, yp, zp = geodetic2geocentricXYZ(gdlatp, glonp, h, returnR = False, degrees = True)

thetap, rp, _, _ = geod2geoc(gdlatp, h, 0., 0.)
gclatp = 90.-thetap

# h2, lat2, lon2 = ECEF2geodetic(xp,yp,zp)
# assert np.all(np.isclose(np.ones_like(lat2)*h,h2))
# assert np.all(np.isclose(lat2,gdlatp))
# assert np.all(np.isclose(lon2,glonp))

# 2. Get B-field unit vectors b_k at points in question

Bp_sph = np.vstack(ppigrf.igrf_gc(rp,thetap,glonp,igrf_refdate))
# Brp, Bthetap, Bphip = ppigrf.igrf_gc(rp,thetap,glonp,igrf_refdate)

# unit B-field vectors in spherical (r, theta, phi) coordinates
bp_sph = Bp_sph/np.linalg.norm(Bp_sph,axis=0)
bp_sphenu = np.stack([bp_sph[2], -bp_sph[1], bp_sph[0]])

# Rvec_SPHENU_TO_ECEF_p = np.transpose(ECEFtosphericalENUMatrix(glonp, gclatp),axes=[1,0,2])

Rvec_SPHENU_TO_ECEF_p = ECEFtosphericalENUMatrix(glonp, gclatp)

bp_ecef = np.einsum('ij...,i...',Rvec_SPHENU_TO_ECEF_p,bp_sphenu).T

# SANITY CHECK
# Rvec_SPHENU_TO_ECEF_1p = ECEFtosphericalENUMatrix(glonp[0], gclatp[0]).squeeze().T
# check = Rvec_SPHENU_TO_ECEF_1p@bp_sphenu[:,0]
# assert np.all(np.isclose(check,bp_ecef[:,0]))

# 3. Get unit vectors l_k that point from Skibotn to points in question
L_ecef = np.stack([xp-xt,yp-yt,zp-zt])
l_ecef = L_ecef/np.linalg.norm(L_ecef,axis=0)

# 4. Find point k for which dot(b_k,l_k) is maximized (or actually minimized)
dot_bl = np.sum(l_ecef*bp_ecef,axis=0)

win_ip = np.argmin(dot_bl)

# 5. Get el, az of this point

# lwin = l_ecef[:,win_ip]
# az0 = np.rad2deg(np.arctan2(np.sum(e_ecef*lwin),np.sum(n_ecef*lwin)))
# eaz0 = np.sin(np.deg2rad(az0))*e_ecef+np.cos(np.deg2rad(az0))*n_ecef
# el0 = np.rad2deg(np.arctan2(np.sum(lwin*u_ecef),np.sum(lwin*eaz0)))

# az02, el02 = get_receiver_az_el(gclat_t,glon_t,np.array([xp[win_ip],yp[win_ip],zp[win_ip]]),R=rt)
# az01, el01 = get_receiver_az_el_geod_ECEF(gdlat_t,glon_t,np.array([xp[win_ip],yp[win_ip],zp[win_ip]]).T)
az0, el0 = get_receiver_az_el_geod(gdlat_t,glon_t,gdlatp[win_ip],glonp[win_ip],h)

##############################
# Pick el,az to go along with master point
# Here we approximate el and az as orthogonal _Cartesian_ coordinates


##############################
## Make figure, show things

fig = plt.figure()
im = plt.contourf(glonp.reshape(dlshape),gdlatp.reshape(dlshape),dot_bl.reshape(dlshape),levels=100)
plt.contour(glonp.reshape(dlshape),gdlatp.reshape(dlshape),dot_bl.reshape(dlshape),colors='k')
# im = plt.imshow(dot_bl.reshape(dlshape),
#                 extent=[glonp.min(),glonp.max(),
#                         gdlatp.min(),gdlatp.max()])
plt.xlabel("Geographic lon [deg]")
plt.ylabel("Geodetic lat [deg]")
cb = plt.colorbar(im)
cb.set_label("dot(bhat,lhat)")
plt.scatter(glon_t, gdlat_t, marker='*',s=20,label="Skibotn",color='C0')
plt.scatter(glonp[win_ip], gdlatp[win_ip], marker='*',s=20,label="min",color='orange')


