"""
This example illustrates how much unique information is contained in the LOS wave vectors from each
EISCAT_3D transmitter-receiver pair given a reference height h.

In the figure produced by this script, the letters A, B, and C denote the following
transmitter-receiver pairs:

A: SKI-SKI
B: SKI-KAI
C: SKI-KRS

Thus, for example, the top left panel shows the angle of the dot product of the wave vectors
resulting from pair A (SKI-SKI) and pair B (SKI-KAI).

In each panel of the figure the locations of the Skibotn, Kaiseniemi, and Karesuvanto sites are
respectively denoted by yellow, orange, and green stars.

The region interior to the thick blue line in each panel denotes where the elevation of both sites
is ≥ 30° (30° is the minimum allowable elevation). This region is obviously a function of height,
which is represented by the variable "h" below. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Why might this figure be useful?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
My idea with this figure was to get a sense of where we we are able (at least based on geometry)
to measure the full ion velocity vector at each altitude. Since what is shown is contours of angles
between the various LOS wave vectors*, the higher the angle, the more unique the information contained
in LOS measurements of the two transmitter-receiver pairs. 

*At least in the top row and the bottom left panel; the bottom right panel shows the _average_ angle
for the three bistatic pairs A, B, and C)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Math Background
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The wave vector is calculated using Eq. 1 in Virtanen et al (2014,doi:10.1002/2014JA020540):

k^s = k^r - k^t,

where 

k^t: wave vector of transmitted signal;
k^r: wave vector of scattered signal entering receiver;
k^s: wave vector of ion acoustic waves from which scattering takes place.

This means, for example, that for bistatic pair A (Skibotn is both transmitter and receiver), the
scattering vector k^s points directly from the scattering point to Skibotn.

Thus with receiver r and transmitter t we measure the projection (Eq. 2 in Virtanen et al, 2014)

v_los = v . k^s / |k^s|

of the true ion velocity vector v.

2024/10/03
SMH
"""

from datetime import datetime
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams.update({'font.size': 12})

plt.ion()

import ppigrf

from lompe import cs
import os

import e3doubt
from e3doubt import experiment

from e3doubt.utils import get_supported_sites
from e3doubt.geodesy import geodetic2geocentricXYZ,ECEF2geodetic

import e3doubt.radar_utils
from importlib import reload
reload(e3doubt.radar_utils)
from e3doubt.radar_utils import get_points_az_el_geod_ECEF, get_range_line, get_kvec

sites = get_supported_sites()

igrf_refdate = datetime(2020,12,1)

gdlat_t, glon_t = sites.loc['SKI']
gdlat_r1, glon_r1 = sites.loc['KAI']
gdlat_r2, glon_r2 = sites.loc['KRS']

xt, yt, zt = geodetic2geocentricXYZ(gdlat_t, glon_t, 0, returnR = False, degrees = True)
xr1, yr1, zr1 = geodetic2geocentricXYZ(gdlat_r1, glon_r1, 0, returnR = False, degrees = True)
xr2, yr2, zr2 = geodetic2geocentricXYZ(gdlat_r2, glon_r2, 0, returnR = False, degrees = True)

##############################
# Calculate measure of where E3D can best estimate full ion drift vector 

## 1. Generate a bunch of points over Skibotn
h = 200                         # km
ddeg = 0.02

# degdimlon = 20.0
degdimlat = 10.0                # ORIG
degdimlat = 5.0                # NEW

dlat = np.arange(-degdimlat,degdimlat,ddeg)

# dlon = np.arange(-degdimlon,degdimlon,ddeg)
# dlat, dlon = np.meshgrid(dlat,dlon,indexing='ij')
# dlat, dlon = dlat.ravel(), dlon.ravel()

glonp = np.arange(0,40,ddeg)    # ORIG
glonp = np.arange(10,30,ddeg)    # NEW
dlat, glonp = np.meshgrid(dlat,glonp,indexing='ij')

dlshape = dlat.shape

dlat, glonp = dlat.ravel(), glonp.ravel()

gdlatp = gdlat_t + dlat
# glonp = glon_t+dlon

xp, yp, zp = geodetic2geocentricXYZ(gdlatp, glonp, h, returnR = False, degrees = True)

#####
# pointing from transmitter to point over E3D 
# dxtp, dytp, dztp = xt-xp, yt-yp, zt-zp

# kt_hat = np.vstack([dxtp,dytp,dztp])
# kt_hat = kt_hat/np.linalg.norm(kt_hat,axis=0)

kt_hat = get_kvec(xt,yt,zt,xp,yp,zp)


#####
# pointing from point over E3D to receiver

# dxpt, dypt, dzpt = xp-xt, yp-yt, zp-zt
# dxpr1, dypr1, dzpr1 = xp-xr1, yp-yr1, zp-zr1
# dxpr2, dypr2, dzpr2 = xp-xr2, yp-yr2, zp-zr2

# get_kvec(xp,yp,zp,xt,yt,zt)

kr1_hat = get_kvec(xp,yp,zp,xr1,yr1,zr1)
kr2_hat = get_kvec(xp,yp,zp,xr2,yr2,zr2)

#####
# Unit k vectors for each transmitter-receiver pair (see e.g. Eq 1 in Virtanen et al, 2014, doi:10.1002/2014JA020540)
#A: SKI-SKI
#B: SKI-KAI
#C: SKI-KRS
ksA = -kt_hat-kt_hat
ksB = kr1_hat-kt_hat
ksC = kr2_hat-kt_hat

ksA = ksA/np.linalg.norm(ksA,axis=0)
ksB = ksB/np.linalg.norm(ksB,axis=0)
ksC = ksC/np.linalg.norm(ksC,axis=0)

checker = lambda x: np.all(np.isclose(np.linalg.norm(x,axis=0),1))
assert checker(ksA) and checker(ksB) and checker(ksC),"Some of your 'unit' vectors are not unit length!"

# Dot products between k vectors
dotAB = np.sum(ksA*ksB,axis=0)
dotAC = np.sum(ksA*ksC,axis=0)
dotBC = np.sum(ksB*ksC,axis=0)

cosAB = np.rad2deg(np.arccos(np.clip(dotAB,-1.,1.)))
cosAC = np.rad2deg(np.arccos(np.clip(dotAC,-1.,1.)))
cosBC = np.rad2deg(np.arccos(np.clip(dotBC,-1.,1.)))

#Find out where elevation is impossible?
min_elevation = 30          # hard number for EISCAT association, I think

azt, elt = get_points_az_el_geod_ECEF(gdlat_t,glon_t,np.vstack([xp,yp,zp]).T,hRec=0.)
azr1, elr1 = get_points_az_el_geod_ECEF(gdlat_r1,glon_r1,np.vstack([xp,yp,zp]).T,hRec=0.)
azr2, elr2 = get_points_az_el_geod_ECEF(gdlat_r2,glon_r2,np.vstack([xp,yp,zp]).T,hRec=0.)

badAB = (elt  < min_elevation) | (elr1 < min_elevation)
badAC = (elt  < min_elevation) | (elr2 < min_elevation)
badBC = (elt  < min_elevation) | (elr1 < min_elevation) | (elr2 < min_elevation)

apply_elevation_screening = False

if apply_elevation_screening:
    replaceme = np.nan
    replaceme = 1

    print(f"About to replace the dot product with {replaceme} where the elevation drops below {min_elevation}° for either site ")
    print(f"Replacing {badAB.sum()} points ({badAB.sum()/len(badAB)*100:.4f}%) for (SKI-SKI)/(SKI-KAI) wave vector pair")
    print(f"Replacing {badAC.sum()} points ({badAC.sum()/len(badAC)*100:.4f}%) for (SKI-SKI)/(SKI-KRS) wave vector pair")
    print(f"Replacing {badBC.sum()} points ({badBC.sum()/len(badBC)*100:.4f}%) for (SKI-KAI)/(SKI-KRS) wave vector pair")

    dotAB[badAB] = replaceme
    dotAC[badAC] = replaceme
    dotBC[badBC] = replaceme


#plot these jerks

##############################
## Make figure, show things

# get coastlines
datapath = os.path.dirname(os.path.abspath(cs.__file__)) + '/data/'
def get_coastlines(resolution = '50m'):
    """ generate coastlines in projected coordinates """

    coastlines = np.load(datapath + 'coastlines_' + resolution + '.npz')
    for key in coastlines:
        lat, lon = coastlines[key]
        yield lon, lat

fig = plt.figure(figsize=(10,10),num=10)
plt.clf()
_ = fig.suptitle(r"$h = $"+f"{h} km")

ncol = 2
nrow = 2
ax00 = plt.subplot(nrow, ncol, 1)
ax01 = plt.subplot(nrow, ncol, 2)#, sharex=ax00,sharey=ax00)
ax02 = plt.subplot(nrow, ncol, 3)#, sharex=ax00,sharey=ax00)
ax03 = plt.subplot(nrow, ncol, 4)#, sharex=ax00,sharey=ax00)

axes = [ax00,ax01,ax02, ax03]

labs = ['a','b','c','d']

for i,ax in enumerate(axes):
    plt.sca(ax)
    plt.xlim((glonp.min(),glonp.max()))
    plt.ylim((gdlatp.min(),gdlatp.max()))

    ax.text(0.07, 0.95, labs[i],
            fontsize='xx-large',
            fontweight='bold',
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes)

txopts = dict(marker='*',s=100,label="Skibotn",color='yellow',edgecolors='k',zorder=1.5)
rxopts = dict(marker='*',s=100,label="Kaiseniemi",color='C1',edgecolors='k',zorder=1.5)
rxopts2 = dict(marker='*',s=100,label="Karesuvanto",color='C2',edgecolors='k',zorder=1.5)
coastopts = dict(color='gray',linewidth=1,zorder=1)
coastres = '10m' # '50m'

def fmt(x):
    s = f"{x:.0f}"
    return rf"{s}$^\circ$" if plt.rcParams["text.usetex"] else f"{s}°"
def fmtzeronodeg(x):
    s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"
def fmtdeg(x):
    s = f"{x:.1f}"
    return rf"{s}$^\circ$" if plt.rcParams["text.usetex"] else f"{s}°"

def fmtnodeg(x):
    s = f"{x:.1f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

## Apex coordinates, 30° elevation line
ax = axes[0]
plt.sca(ax)
ax.set_title(r"$\cos^{-1}\left(\mathbf{k}_A \cdot \mathbf{k}_B\right)$"+" [deg]")
# ax.set_title("$\cos^{-1}\left(\mathbf{k}_A \cdot \mathbf{k}_B\right)$ [deg]")
# plt.xlabel("Geographic lon [deg]")
plt.ylabel("Geodetic lat [deg]")

GLON,GDLAT = glonp.reshape(dlshape),gdlatp.reshape(dlshape)

### Show where we can see stuff
imvisiblecolor = 'b'
imvisiblecolor = 'C0'
imvisiblelw = 3
imvisibleAB = ax.contour(GLON,GDLAT,badAB.reshape(dlshape),colors=imvisiblecolor,linewidths=imvisiblelw)


for cllon,cllat in get_coastlines(resolution=coastres):
    plt.plot(cllon,cllat,**coastopts)

### Show dot product of k vectors

if h < 200:
    dotlevels = [5,10,15,20,25,30,40,50]
elif (h >= 200) & (h < 300):
    dotlevels = [3,6,9,12,15]
    # dotlevels = [2.5,5,7.5,10,12.5,15.]
    # dotlevels = [2,4,6,8,10,12,14]
elif (h >= 300) & (h < 400):
    dotlevels = [2,4,6,8,10,12,14]


# dotABlevels = 
# inputs = [GLON,GDLAT]
add_levels = True
dotlevels = dotlevels if add_levels else None
# if add_levels:
#     inputs.append(dotlevels)
imdotAB = ax.contour(GLON,GDLAT,cosAB.reshape(dlshape),dotlevels,colors='k')
ax.clabel(imdotAB, imdotAB.levels, colors='k', inline=1, fontsize=12)

### 30° elevation ring
# ringaz = np.arange(0,361,1)
# ringel = np.ones_like(ringaz)*30.
# ringheights = [100,300,500]
# rings = [ECEF2geodetic(*get_range_line(gdlat_t, glon_t, ringaz, ringel, h).T) for h in ringheights]

# ringcols = ['C0']*3
# ringlw = [1,2,3]
# for i,ring in enumerate(rings):
#     ringh, ringlat, ringlon = ring

#     plt.plot(ringlon,ringlat,lw=ringlw[i],color=ringcols[i])

### E3D sites
plt.scatter(glon_t, gdlat_t, **txopts)
plt.scatter(glon_r1, gdlat_r1, **rxopts)
plt.scatter(glon_r2, gdlat_r2, **rxopts2)
# plt.scatter(glonp[win_ip], gdlatp[win_ip], marker='^',s=25,label="min",color='gray',edgecolor='k',zorder=100,alpha=.8)

plt.xlim((glonp.min(),glonp.max()))
plt.ylim((gdlatp.min(),gdlatp.max()))
plt.xticks(plt.xticks()[0][1:-1],labels=[" " for x in plt.xticks()[1][1:-1]])

plt.grid(color='gray',lw=0.5)

####################
# now AC
ax = axes[1]
plt.sca(ax)
ax.set_title(r"$\cos^{-1}\left(\mathbf{k}_A \cdot \mathbf{k}_C\right)$"+" [deg]")
# plt.xlabel("Geographic lon [deg]")
# plt.ylabel("Geodetic lat [deg]")

for cllon,cllat in get_coastlines(resolution=coastres):
    plt.plot(cllon,cllat,**coastopts)

### Show where we can see stuff
imvisibleAC = ax.contour(GLON,GDLAT,badAC.reshape(dlshape),colors=imvisiblecolor,linewidths=imvisiblelw)


imdotAC = ax.contour(GLON,GDLAT,cosAC.reshape(dlshape),dotlevels,colors='k')
ax.clabel(imdotAC, imdotAC.levels, colors='k', inline=1, fontsize=12)

### E3D sites
plt.scatter(glon_t, gdlat_t, **txopts)
plt.scatter(glon_r1, gdlat_r1, **rxopts)
plt.scatter(glon_r2, gdlat_r2, **rxopts2)
# plt.scatter(glonp[win_ip], gdlatp[win_ip], marker='^',s=25,label="min",color='gray',edgecolor='k',zorder=100,alpha=.8)

plt.xlim((glonp.min(),glonp.max()))
plt.ylim((gdlatp.min(),gdlatp.max()))

plt.yticks(plt.yticks()[0][1:-1],labels=[" " for x in plt.yticks()[1][1:-1]])
plt.xticks(plt.xticks()[0][1:-1],labels=[" " for x in plt.xticks()[1][1:-1]])

plt.grid(color='gray',lw=0.5)

####################
# now BC
ax = axes[2]
plt.sca(ax)
ax.set_title(r"$\cos^{-1}\left(\mathbf{k}_B \cdot \mathbf{k}_C\right)$"+" [deg]")
plt.xlabel("Geographic lon [deg]")
plt.ylabel("Geodetic lat [deg]")

for cllon,cllat in get_coastlines(resolution=coastres):
    plt.plot(cllon,cllat,**coastopts)

### Show where we can see stuff
imvisibleBC = ax.contour(GLON,GDLAT,badBC.reshape(dlshape),colors=imvisiblecolor,linewidths=imvisiblelw)

# dotABlevels = 
imdotBC = ax.contour(GLON,GDLAT,cosBC.reshape(dlshape),dotlevels,colors='k')
ax.clabel(imdotBC, imdotBC.levels, colors='k', inline=1, fontsize=12)

### E3D sites
plt.scatter(glon_t, gdlat_t, **txopts)
plt.scatter(glon_r1, gdlat_r1, **rxopts)
plt.scatter(glon_r2, gdlat_r2, **rxopts2)
# plt.scatter(glonp[win_ip], gdlatp[win_ip], marker='^',s=25,label="min",color='gray',edgecolor='k',zorder=100,alpha=.8)

plt.xlim((glonp.min(),glonp.max()))
plt.ylim((gdlatp.min(),gdlatp.max()))
# plt.xticks(plt.xticks()[0][1:-1],labels=[" " for x in plt.xticks()[1][1:-1]])

plt.grid(color='gray',lw=0.5)

####################
# now all
ax = axes[3]
plt.sca(ax)
ax.set_title(r"$\frac{1}{3}\sum_i \sum_{j>i} \cos^{-1}\left(\mathbf{k}_i \cdot \mathbf{k}_j\right)$"+" [deg]")
plt.xlabel("Geographic lon [deg]")
# plt.ylabel("Geodetic lat [deg]")

for cllon,cllat in get_coastlines(resolution=coastres):
    plt.plot(cllon,cllat,**coastopts)

### Show where we can see stuff
imvisibleBC = ax.contour(GLON,GDLAT,badBC.reshape(dlshape),colors=imvisiblecolor,linewidths=imvisiblelw)


# dotABlevels = 
imdotall = ax.contour(GLON,GDLAT,(cosAB+cosAC+cosBC).reshape(dlshape)/3,dotlevels,colors='k')
ax.clabel(imdotall, imdotall.levels, colors='k', inline=1, fontsize=12)

### E3D sites
plt.scatter(glon_t, gdlat_t, **txopts)
plt.scatter(glon_r1, gdlat_r1, **rxopts)
plt.scatter(glon_r2, gdlat_r2, **rxopts2)
# plt.scatter(glonp[win_ip], gdlatp[win_ip], marker='^',s=25,label="min",color='gray',edgecolor='k',zorder=100,alpha=.8)


plt.xlim((glonp.min(),glonp.max()))
plt.ylim((gdlatp.min(),gdlatp.max()))
plt.yticks(plt.yticks()[0][1:-1],labels=[" " for x in plt.yticks()[1][1:-1]])

plt.grid(color='gray',lw=0.5)

plt.tight_layout()

plt.savefig(f"./plots/7__degeneracy_of_los_vectors_of_e3d__h={h}km.png")
