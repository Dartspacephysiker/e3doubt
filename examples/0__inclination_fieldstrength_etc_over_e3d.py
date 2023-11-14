from datetime import datetime
import numpy as np

from matplotlib import pyplot as plt
plt.ion()

import ppigrf

import e3doubt
from importlib import reload
reload(e3doubt)
from e3doubt import experiment

from e3doubt.utils import get_supported_sites,field_geometry_lonlat
from e3doubt.geodesy import geodetic2geocentricXYZ,ECEF2geodetic,geod2geoc

import e3doubt.radar_utils
reload(e3doubt.radar_utils)
from e3doubt.radar_utils import get_enu_vectors_cartesian,get_receiver_az_el_geod,ECEFtosphericalENUMatrix,get_range_line

sites = get_supported_sites()

igrf_refdate = datetime(2020,12,1)
apex_refh = 110

gdlat_t, glon_t = sites.loc['SKI']
gdlat_r1, glon_r1 = sites.loc['KAI']
gdlat_r2, glon_r2 = sites.loc['KRS']
xt, yt, zt = geodetic2geocentricXYZ(gdlat_t, glon_t, 0, returnR = False, degrees = True)

##############################
# Find az, el where Skibotn transceiver aligns best with stuff

# 1. Generate a bunch of points over Skibotn
h = 300                         # km
ddeg = 0.1
degdimlat = 10.0
degdimlon = 20.0
dlat = np.arange(-degdimlat,degdimlat,ddeg)
dlon = np.arange(-degdimlon,degdimlon,ddeg)

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
magBp = np.linalg.norm(Bp_sph,axis=0)
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

angler = np.rad2deg(np.arccos(dot_bl))
# angler[angler > 90] = 90-angler
angler = angler.reshape(dlshape)

# 5. Get el, az of this point

# lwin = l_ecef[:,win_ip]
# az0 = np.rad2deg(np.arctan2(np.sum(e_ecef*lwin),np.sum(n_ecef*lwin)))
# eaz0 = np.sin(np.deg2rad(az0))*e_ecef+np.cos(np.deg2rad(az0))*n_ecef
# el0 = np.rad2deg(np.arctan2(np.sum(lwin*u_ecef),np.sum(lwin*eaz0)))

# az02, el02 = get_receiver_az_el(gclat_t,glon_t,np.array([xp[win_ip],yp[win_ip],zp[win_ip]]),R=rt)
# az01, el01 = get_receiver_az_el_geod_ECEF(gdlat_t,glon_t,np.array([xp[win_ip],yp[win_ip],zp[win_ip]]).T)
az0, el0 = get_receiver_az_el_geod(gdlat_t,glon_t,gdlatp[win_ip],glonp[win_ip],h)

incp, decp, diplatp = field_geometry_lonlat(glonp, gdlatp, igrf_refdate, h_km=apex_refh)

##############################
## Make figure, show things

# get coastlines
from lompe import cs
import os
datapath = os.path.dirname(os.path.abspath(cs.__file__)) + '/data/'
def get_coastlines(resolution = '50m'):
    """ generate coastlines in projected coordinates """

    coastlines = np.load(datapath + 'coastlines_' + resolution + '.npz')
    for key in coastlines:
        lat, lon = coastlines[key]
        yield lon, lat

fig = plt.figure(figsize=(18,6),num=10)
plt.clf()

ncol = 4
nrow = 1
ax00 = plt.subplot(nrow, ncol, 1)
ax01 = plt.subplot(nrow, ncol, 2)#, sharex=ax00,sharey=ax00)
ax02 = plt.subplot(nrow, ncol, 3)#, sharex=ax00,sharey=ax00)
ax03 = plt.subplot(nrow, ncol, 4)#, sharex=ax00,sharey=ax00)

axes = [ax00,ax01,ax02, ax03]

for ax in axes:
    plt.sca(ax)
    plt.xlim((glonp.min(),glonp.max()))
    plt.ylim((gdlatp.min(),gdlatp.max()))


txopts = dict(marker='*',s=100,label="Skibotn",color='yellow',edgecolors='k',zorder=100)
rxopts = dict(marker='*',s=100,label="Kaiseniemi",color='C1',edgecolors='k',zorder=100)
coastopts = dict(color='gray',linewidth=1,zorder=1)
coastres = '10m' # '50m'

def fmt(x):
    s = f"{x:.0f}"
    return rf"{s}$^\circ$" if plt.rcParams["text.usetex"] else f"{s}°"
def fmtinc(x):
    s = f"{x:.1f}"
    return rf"{s}$^\circ$" if plt.rcParams["text.usetex"] else f"{s}°"

## Apex coordinates, 30° elevation line
ax = axes[0]
plt.sca(ax)
ax.set_title("Modified Apex-110 coordinates")
plt.xlabel("Geographic lon [deg]")
plt.ylabel("Geodetic lat [deg]")

from apexpy import Apex
a = Apex(igrf_refdate,apex_refh)

alatp, alonp = a.geo2apex(gdlatp,glonp,apex_refh)

# 30° elevation ring
ringaz = np.arange(0,361,1)
ringel = np.ones_like(ringaz)*30.
ringheights = [100,300,500]
rings = [ECEF2geodetic(*get_range_line(gdlat_t, glon_t, ringaz, ringel, h).T) for h in ringheights]

imclat = ax.contour(glonp.reshape(dlshape),gdlatp.reshape(dlshape),alatp.reshape(dlshape),colors='k')
imclon = ax.contour(glonp.reshape(dlshape),gdlatp.reshape(dlshape),alonp.reshape(dlshape),colors='k')
ax.clabel(imclat, imclat.levels, fmt=fmt, colors='k', inline=1, fontsize=10)
ax.clabel(imclon, imclon.levels, fmt=fmt, colors='k', inline=1, fontsize=10)

plt.scatter(glon_t, gdlat_t, **txopts)
plt.scatter(glon_r1, gdlat_r1, **rxopts)
plt.scatter(glon_r2, gdlat_r2, **rxopts)
plt.scatter(glonp[win_ip], gdlatp[win_ip], marker='^',s=40,label="min",color='black',zorder=100)

ringcols = ['C0']*3
ringlw = [1,2,3]
for i,ring in enumerate(rings):
    ringh, ringlat, ringlon = ring

    plt.plot(ringlon,ringlat,lw=ringlw[i],color=ringcols[i])


for cllon,cllat in get_coastlines(resolution=coastres):
    plt.plot(cllon,cllat,**coastopts)

plt.xlim((glonp.min(),glonp.max()))
plt.ylim((gdlatp.min(),gdlatp.max()))

## Angle between Skibotn LOS vector and bhat
ax = axes[1]
plt.sca(ax)
# ax.set_title("angle(bhat,lhat)")
ax.set_title(r"$\hat{\mathbf{b}}\cdot\hat{\mathbf{l}}$")
plt.xlabel("Geographic lon [deg]")
# im = ax.contourf(glonp.reshape(dlshape),gdlatp.reshape(dlshape),angler,levels=100,cmap=plt.get_cmap('viridis_r'))
imc = ax.contour(glonp.reshape(dlshape),gdlatp.reshape(dlshape),angler,colors='k')
# im = plt.imshow(dot_bl.reshape(dlshape),
#                 extent=[glonp.min(),glonp.max(),
#                         gdlatp.min(),gdlatp.max()])
ax.clabel(imc, imc.levels, fmt=fmt, colors='k', inline=1, fontsize=10)
# cb = plt.colorbar(im,location='bottom')
# cb.set_label("")
plt.scatter(glon_t, gdlat_t, **txopts)
plt.scatter(glon_r1, gdlat_r1, **rxopts)
plt.scatter(glon_r2, gdlat_r2, **rxopts)
plt.scatter(glonp[win_ip], gdlatp[win_ip], marker='^',s=40,label="min",color='black',zorder=100)

for cllon,cllat in get_coastlines(resolution=coastres):
    plt.plot(cllon,cllat,**coastopts)

plt.xlim((glonp.min(),glonp.max()))
plt.ylim((gdlatp.min(),gdlatp.max()))
plt.yticks(plt.yticks()[0][1:-1],labels=[" " for x in plt.yticks()[1][1:-1]])


## Field strength over E3D
ax = axes[2]
plt.sca(ax)
ax.set_title("Main field strength [10$^4$ nT]")
plt.xlabel("Geographic lon [deg]")
# lablevels = np.array([4.67, 4.68, 4.69, 4.7, 4.71, 4.72, 4.73, 4.74, 4.75, 4.76, 4.77, 4.78, 4.79, 4.8, 4.81])
# lablevels = np.arange(np.round(magBp.min()/1e4,2),magBp.max()/1e4,0.05)
# imm = ax.contourf(glonp.reshape(dlshape),gdlatp.reshape(dlshape),magBp.reshape(dlshape)/1e4,levels=100)
# immc = ax.contour(glonp.reshape(dlshape),gdlatp.reshape(dlshape),magBp.reshape(dlshape)/1e4,levels=lablevels[::2],colors='k')
immc = ax.contour(glonp.reshape(dlshape),gdlatp.reshape(dlshape),magBp.reshape(dlshape)/1e4,colors='k')
plt.scatter(glon_t, gdlat_t, **txopts)
plt.scatter(glon_r1, gdlat_r1, **rxopts)
plt.scatter(glon_r2, gdlat_r2, **rxopts)
# cbm = plt.colorbar(imm,location='bottom')
# cbm.set_label("nT")

ax.clabel(immc, immc.levels, colors='k', inline=1, fontsize=10)

for cllon,cllat in get_coastlines(resolution=coastres):
    plt.plot(cllon,cllat,**coastopts)

plt.xlim((glonp.min(),glonp.max()))
plt.ylim((gdlatp.min(),gdlatp.max()))
plt.yticks(plt.yticks()[0][1:-1],labels=[" " for x in plt.yticks()[1][1:-1]])


## Inclination over E3D
ax = axes[3]
plt.sca(ax)
ax.set_title("Main field inclination")
plt.xlabel("Geographic lon [deg]")
# im2 = ax.contourf(glonp.reshape(dlshape),gdlatp.reshape(dlshape),incp.reshape(dlshape),levels=100)
# immc = ax.contour(glonp.reshape(dlshape),gdlatp.reshape(dlshape),magBp.reshape(dlshape)/1e4,levels=lablevels[::2],colors='k')
# inclevels = np.arange(76,80.6,0.5)
im2c = ax.contour(glonp.reshape(dlshape),gdlatp.reshape(dlshape),incp.reshape(dlshape),colors='k')#,levels=inclevels)
plt.scatter(glon_t, gdlat_t, **txopts)
plt.scatter(glon_r1, gdlat_r1, **rxopts)
plt.scatter(glon_r2, gdlat_r2, **rxopts)

ax.clabel(im2c, im2c.levels, fmt=fmtinc, colors='k', inline=1, fontsize=10)

for cllon,cllat in get_coastlines(resolution=coastres):
    plt.plot(cllon,cllat,**coastopts)

plt.xlim((glonp.min(),glonp.max()))
plt.ylim((gdlatp.min(),gdlatp.max()))
plt.yticks(plt.yticks()[0][1:-1],labels=[" " for x in plt.yticks()[1][1:-1]])

plt.tight_layout()

plt.savefig("./plots/0__inclination_fieldstrength_etc_over_e3d.png")
