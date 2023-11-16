from datetime import datetime
import numpy as np

from matplotlib import pyplot as plt
plt.ion()

import ppigrf

from e3doubt import experiment

from e3doubt.utils import get_supported_sites
from e3doubt.geodesy import geodetic2geocentricXYZ,ECEF2geodetic,geod2geoc

from e3doubt.radar_utils import get_2D_csgrid_az_el

from apexpy import Apex

def gridedges(lon,lat):
    #bottom edge
    b = (slice(None,None,None),0)
    r = (-1,slice(None,None,None))
    t = (slice(None,None,-1),-1)
    l = (0,slice(None,None,-1))

    lonbound = np.concatenate([lon[b],lon[r],lon[t],lon[l]])
    latbound = np.concatenate([lat[b],lat[r],lat[t],lat[l]])

    return lonbound,latbound


sites = get_supported_sites()

# Get transceiver and receiver locations
gdlat_t, glon_t = sites.loc['SKI']
gdlat_r1, glon_r1 = sites.loc['KAI']
gdlat_r2, glon_r2 = sites.loc['KRS']

# Define grid
h_grid = 200
Lgrid = 100e3
Wgrid = 100e3
Lresgrid = Wresgrid = 10e3 #grid resolution in m                                 
az, el, gdlatg, glong, h, gridm = get_2D_csgrid_az_el(gdlat_t, glon_t, h=h_grid,
                                                      L=Lgrid, W=Wgrid,
                                                      Lres=Lresgrid, Wres=Wresgrid,
                                                      return_grid=True)

H  = np.array([200,220])

exp = experiment.Experiment(az=az,el=el,h=H)
exp.run_models()
dfunc = exp.calc_uncertainties(integrationsec=300)

points = exp.get_points()
radarconfig = exp.get_radarconfig()
Npt = exp.N['pt']

igrf_refdate = datetime(2020,12,1)

apex_refh = 110
a = Apex(igrf_refdate,refh=apex_refh)

gdlat, glon, h = points['gdlat'].values,points['glon'].values, points['h'].values

# Get gdlat and glon of points at ionospheric altitude
alat, alon = a.geo2apex(gdlat, glon, h)

gdlatp, glonp, err = a.apex2geo(alat, alon, apex_refh)

## 1. Get velocity covariance matrices

covmats = exp.get_velocity_cov_matrix()

## 2. Project velocity covariance into covariance of matrices at ionospheric height
# This is done using Apex basis vectors

# get base vectors at height and at reference altitude, where 'p' mean 'prime' (i.e., at reference altitude)
# components are geodetic east, north, and up
f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = a.basevectors_apex(gdlat, glon, h)
f1p, f2p, f3p, g1p, g2p, g3p, d1p, d2p, d3p, e1p, e2p, e3p = a.basevectors_apex(gdlatp, glonp, apex_refh)

e1pe2p = np.transpose(np.stack([e1p,e2p]),axes=[1, 0, 2])  # dimension 0: ENU components; 1: e1p or e2p; 2: meas location
d1d2 = np.stack([d1,d2]) # dimension 0: d1 or d2; 1: ENU components; 2: meas location

# Form B matrix, which represents calculation of ve1 (=dot(v,d1)) and ve2(=dot(v,d1)) followed by projection to reference ionospheric height 
# sum is over e1p/d1 and e2p/d2 vectors, so first two dimensions of B both index ENU components. last dimension is meas location
B = np.transpose(np.einsum('ij...,jk...',e1pe2p,d1d2),axes=[1,2,0])  # transpose to put measurement index as last index

# Project velocity vector covariance at measurement height into (perpendicular-to-B-)velocity-vector-at-reference-ionospheric-height space
covp = np.transpose(np.einsum('ij...,jk...,lk...',B,covmats,B),axes=[1,2,0])  # use 'lk' and not 'kl' at the end because we need transpose of B, not B. 

## 3. Select model for velocity 
# Let's go for div-free SECS to begin with

# set up cubed sphere projection and grid
import lompe
from lompe import cs
from secsy import CSplot
# sites = get_supported_sites()

tx = radarconfig['tx']
gclat_t, gdlat_t, glon_t = tx['gclat'], tx['gdlat'], tx['glon']
RE = 6371.2e3
RI = RE+110e3
L = 200e3
W = 200e3
Lres = Wres = 20e3
projection = cs.CSprojection((glon_t, gclat_t), 0) # central (lon, lat) and orientation of the grid
grid = cs.CSgrid(projection, L, W, Lres, Wres, R = RI, wshift = 0)

xi_t, eta_t = grid.projection.geo2cube(glon_t, gclat_t)

xi_p, eta_p = grid.projection.geo2cube(points['glon'].values, points['gclat'].values)

## Set up lompe model object so that we can rip out convection matrices and stuff

Kp   = 4     # for Hardy model
F107 = 100   # for EUV conductance

# functions for conductances to be passed to the Lompe model
SH = lambda lon = grid.lon, lat = grid.lat: lompe.conductance.hardy_EUV(lon, lat, Kp, time, 'hall')
SP = lambda lon = grid.lon, lat = grid.lat: lompe.conductance.hardy_EUV(lon, lat, Kp, time, 'pedersen')

v_e = np.zeros(Npt)
v_n = np.zeros(Npt)

v_tot = np.vstack((v_e.flatten(), v_n.flatten()))

v_coords = np.vstack((points['glon'].values, points['gclat'].values)) # E-field is defined on the grid
components = [0,1]                                           # eastward, northward components
error=1e1                                                    # m/s
iweight=1.
v_data = lompe.Data(v_tot, v_coords, datatype='convection', components=components, iweight=iweight, error=error)

# initialize model
model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH, SP))

# add input data
model.add_data(v_data)

ds = model.data['convection'][0]
Gs = model.matrix_func['convection'](**ds.coords)
G = np.vstack([G_ for i, G_ in enumerate(Gs) if i in ds.components])

var_ve = covp[0,0,:]
var_vn = covp[1,1,:]
covar_vevn = covp[0,1,:]
covar_vnve = covp[1,0,:]

assert np.all(np.isclose(covar_vevn,covar_vnve))  # sanity check

## Form data covariance matrix
Neval = np.prod(grid.lon.shape)
Cd = np.zeros((Npt*2,Npt*2))

Cd[:Npt,:Npt] = np.diag(var_ve)# upper left, corresponds to all-east components
Cd[Npt:,Npt:] = np.diag(var_vn)# lower right, corresponds to all-north components
Cd[:Npt,Npt:] = np.diag(covar_vevn)# upper right, corresponds to north-east covariance
Cd[Npt:,:Npt] = np.diag(covar_vevn)# lower left, corresponds to east-north covariance

Cdinv = np.linalg.inv(Cd)

GTG = G.T@Cdinv@G

# Applying regularization?
l1 = 0.
# l1 = 0.00001
# l1 = 0.1
LTL = l1 * np.median(GTG)*np.diag(np.ones(GTG.shape[0]))

if not np.isclose(l1,0.):
    print(f"Regularizing result with l1={l1}")

Cpm = np.linalg.inv(GTG+LTL)

Gobss = model.matrix_func['convection'](lon=grid.lon,lat=grid.lat)
Gobs = np.vstack([G_ for i, G_ in enumerate(Gobss) if i in ds.components])
Cpd = Gobs@Cpm@Gobs.T

vm_e_var = np.diag(Cpd[:Neval,:Neval])   # eastward v-model variance
vm_n_var = np.diag(Cpd[Neval:,Neval:])   # northward v-model variance


##############################
## Make a plot

fig = plt.figure(figsize = (14, 10),num=10)
plt.clf()

figtit = r"$\lambda_1$ = "+f"{l1:.5e}"
fig.suptitle(figtit)

axes = []
nrows = 2
ncols = 2
for i in range(nrows*ncols):
    ax = plt.subplot(nrows, ncols, i+1)
    axes.append(ax)

axes = np.array(axes)

axes = axes.ravel()
csaxes = [CSplot(ax, grid, gridtype = 'cs') for ax in axes]

for csi,csax in enumerate(csaxes):
    csax.scatter(glon_t, gclat_t, marker='*',s=200,color='C0',label=tx['name'] if csi == 0 else None,
                 zorder=20)
    for irx in range(exp.N['rx']):
        rx = radarconfig['rx'][irx]
        if rx['name'] == tx['name']:
            continue
        csax.scatter(rx['glon'], rx['gclat'], marker='*',s=150,color='C1',label=rx['name'] if csi == 0 else None,
                     zorder=20)

        
csaxes[0].scatter(points['glon'].values, points['gclat'].values,
                  marker='^', s=10,
                  zorder=21,
                  color='C2',
                  label='obs')

csaxes[1].scatter(grid.lon, grid.lat,
                  marker='s', s=10,
                  zorder=21,
                  color='C3',
                  label='poles')
csaxes[1].scatter(model.grid_E.lon, model.grid_E.lat,
                  marker='o', s=10,
                  zorder=21,
                  color='C4',
                  label='eval pts')

axes[0].legend()
axes[1].legend()

levels = np.arange(-1000,1001,100)
levels = np.array([-3000,-1000,-400,-100,-10,10,100,400,1000,5000])
pcmeshlevels = np.array([-1000,1000])
poslevels = np.array([10,40,100,225,400,1000,5000])
extend = 'both'

axes[2].set_title("Model $v_e$ variance")
axes[3].set_title("Model $v_n$ variance")

# cmap = plt.get_cmap('bwr')
cmap = plt.get_cmap('bwr').resampled(10)
cvme = csaxes[2].pcolormesh(grid.lon, grid.lat, vm_e_var.reshape(grid.lon.shape),
                            cmap=cmap,vmin=pcmeshlevels[0],vmax=pcmeshlevels[-1])
cvmn = csaxes[3].pcolormesh(grid.lon, grid.lat, vm_n_var.reshape(grid.lon.shape),
                          cmap=cmap,vmin=pcmeshlevels[0],vmax=pcmeshlevels[-1])


for csax in csaxes:
    csax.ax.set_xlim((-0.02,0.02))
    csax.ax.set_ylim((-0.02,0.02))

cbe = plt.colorbar(cvme) 
cbn = plt.colorbar(cvmn)

glonb, gclatb = gridedges(gridm.lon,gridm.lat)
for csax in csaxes[2:]:
    csax.plot(glonb,gclatb,color='k',alpha=0.5)


HIGHRES = True
for cl in grid.projection.get_projected_coastlines(resolution='10m' if HIGHRES else '50m'):
    clo, cla = grid.projection.cube2geo(cl[0], cl[1])

    xis, etas = gridm.projection.geo2cube(clo, cla) 
    for csax in csaxes:
        csaxes[0].plot(clo, cla, color='black', linewidth=1)
