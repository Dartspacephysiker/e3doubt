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
from e3doubt.geodesy import geodetic2geocentricXYZ,ECEF2geodetic,geod2geoc,geoc2geod,geodeticheight2geocentricR

import e3doubt.radar_utils
reload(e3doubt.radar_utils)
from e3doubt.radar_utils import get_point_az_el_geod,get_2D_csgrid_az_el, get_field_aligned_beam

from apexpy import Apex

from lompe import cs

import os

# get grid edges
def gridedges(lon,lat):
    #bottom edge
    b = (slice(None,None,None),0)
    r = (-1,slice(None,None,None))
    t = (slice(None,None,-1),-1)
    l = (0,slice(None,None,-1))

    lonbound = np.concatenate([lon[b],lon[r],lon[t],lon[l]])
    latbound = np.concatenate([lat[b],lat[r],lat[t],lat[l]])

    return lonbound,latbound


# get coastlines
datapath = os.path.dirname(os.path.abspath(cs.__file__)) + '/data/'
def get_coastlines(resolution = '50m'):
    """ generate coastlines in projected coordinates """

    coastlines = np.load(datapath + 'coastlines_' + resolution + '.npz')
    for key in coastlines:
        lat, lon = coastlines[key]
        yield lon, lat


radarconfig = experiment.Experiment().get_radarconfig()
sites = get_supported_sites()

gdlat_t, glon_t = sites.loc['SKI']
gdlat_r1, glon_r1 = sites.loc['KAI']
gdlat_r2, glon_r2 = sites.loc['KRS']

## Get equally spaced grid points in cubedsphere coordinates

# set up cubed sphere projection and grid

tx = radarconfig['tx']
gclat_t, gdlat_t, glon_t = tx['gclat'], tx['gdlat'], tx['glon']
h_grid = 200
# RI = geodeticheight2geocentricR(gdlat_t, h_grid)*1e3

# RE = 6371.2e3
# RI = RE+110e3

showgrid = 'small1'
showgrid = 'med1'

if showgrid == 'small1':
    L = 30e3
    W = 30e3
    Lres = Wres = 5e3 #grid resolution in m                                 
elif showgrid == 'med1':
    L = 100e3
    W = 100e3
    Lres = Wres = 10e3 #grid resolution in m                                 
else:
    assert 2<0,"invalid 'wantgrid' selection"

# az1,el1, gdlat1, glon1 = get_field_aligned_beam(h,gdlat_tx=gdlat_t,glon_tx=glon_t,
#                                                 ddeg=0.05,
#                                                 degdimlat=10.0,
#                                                 degdimlon=20.0,
# )

az, el, gdlat, glon, h, grid = get_2D_csgrid_az_el(gdlat_t, glon_t, h_grid=h_grid,
                                                   L=L, W=W,
                                                   Lres=Lres, Wres=Wres,
                                                   return_grid=True,
                                                   ctr_on_fieldaligned_beam=True)

# projection = cs.CSprojection((glon_t, gclat_t), 0) 
# grid = cs.CSgrid(projection, L, W, Lres, Wres, R = RI, wshift = 0)

# xi_t, eta_t = grid.projection.geo2cube(glon_t, gclat_t)

# gclat, glon = grid.lat.ravel(), grid.lon.ravel()
# gdlat, h, _, _ = geoc2geod(90.-gclat, grid.R/1e3, 0., 0.)
# npt = gdlat.size

# az, el = np.zeros(npt), np.zeros(npt)

# for i,(gdla, glo, hi) in enumerate(zip(gdlat,glon,h)):
#     # print(i,gdla,glo)
    
#     azt, elt = get_point_az_el_geod(gdlat_t,glon_t,gdla,glo,hi,hRec=0.)

#     az[i] = azt
#     el[i] = elt


##############################
#Make fig

txopts = dict(marker='*',s=100,label="Skibotn",color='yellow',edgecolors='k')
rxopts = dict(marker='*',s=100,label="Kaiseniemi",color='C1',edgecolors='k')
coastopts = dict(color='gray',linewidth=1)

glonlim = [5,35]
dlat = 5
glatlim = [gdlat_t-dlat,gdlat_t+dlat]
coastres = '50m'

glonlimz = [18,23]
glatlimz = [68,70]

fig = plt.figure(figsize=(11,5.5),num=10)
plt.clf()

ncol = 2
nrow = 1
ax00 = plt.subplot(nrow, ncol, 1)
ax01 = plt.subplot(nrow, ncol, 2)#, sharex=ax00,sharey=ax00)
# ax02 = plt.subplot(nrow, ncol, 3)#, sharex=ax00,sharey=ax00)
# ax03 = plt.subplot(nrow, ncol, 4)#, sharex=ax00,sharey=ax00)

axes = [ax00,ax01]

for ax in axes:
    plt.sca(ax)

    plt.scatter(glon_t, gdlat_t, **txopts)
    plt.scatter(glon_r1, gdlat_r1, **rxopts)
    plt.scatter(glon_r2, gdlat_r2, **rxopts)

# Zoom-out
plt.sca(axes[0])
plt.xlim(glonlim)
plt.ylim(glatlim)

glonb, gdlatb = gridedges(glon.reshape(grid.lat.shape),gdlat.reshape(grid.lat.shape))
plt.plot(glonb,gdlatb,color='C0')
plt.xlabel("Geographic longitude [deg]")
plt.ylabel("Geodetic latitude [deg]")
for cllon,cllat in get_coastlines(resolution='50m'):
    plt.plot(cllon,cllat,**coastopts)


# Zoomed-in view
plt.sca(axes[1])
plt.xlim(glonlimz)
plt.ylim(glatlimz)

plt.scatter(glon,gdlat,marker='o',color='C0')
plt.xlabel("Geographic longitude [deg]")
for cllon,cllat in get_coastlines(resolution='10m'):
    plt.plot(cllon,cllat,**coastopts)
