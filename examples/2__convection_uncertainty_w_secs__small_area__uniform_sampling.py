from datetime import datetime
import numpy as np

from matplotlib import pyplot as plt
plt.ion()

import ppigrf

from e3doubt import experiment

from e3doubt.utils import get_supported_sites
from e3doubt.geodesy import geodetic2geocentricXYZ,ECEF2geodetic,geod2geoc,geodetic2geocentriclat

import e3doubt.radar_utils
from importlib import reload
reload(e3doubt.radar_utils)
from e3doubt.radar_utils import get_2D_csgrid_az_el

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
wshiftgrid = -Wresgrid//2
# wshiftgrid = 0.
az, el, gdlatg, glong, h, gridm = get_2D_csgrid_az_el(gdlat_t, glon_t, h_grid=h_grid,
                                                      L=Lgrid, W=Wgrid,
                                                      Lres=Lresgrid, Wres=Wresgrid,
                                                      wshift=wshiftgrid,
                                                      return_grid=True)

H  = np.array([200,220])

exp = experiment.Experiment(az=az,el=el,h=H)
exp.run_models()
dfunc = exp.get_uncertainties(integrationsec=300)

points = exp.get_points()
radarconfig = exp.get_radarconfig()
Npt = exp.N['pt']

igrf_refdate = datetime(2020,12,1)
apex_refh = 110

gdlat, glon, h = points['gdlat'].values,points['glon'].values, points['h'].values

# Get matrix that lets us map covariance to ionospheric altitude
B, (gdlatmap, glonmap) = get_perpendicular_velocity_mapping_matrix(gdlat,glon,h,
                                                               apex_refh=apex_refh,
                                                               refdate=igrf_refdate,
                                                               return_mapped_coordinates=True)

gclatmap = geodetic2geocentriclat(gdlatmap)

## 1. Get velocity covariance matrices

covmats = exp.get_velocity_cov_matrix()

## 2. Project velocity covariance into covariance of matrices at ionospheric height
# This is done using Apex basis vectors

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
Lres = Wres = 5e3
wshift = -Wres//2
# wshift = 0.
projection = cs.CSprojection((glon_t, gclat_t), 0) # central (lon, lat) and orientation of the grid
grid = cs.CSgrid(projection, L, W, Lres, Wres, R = RI, wshift = wshift)

xi_t, eta_t = grid.projection.geo2cube(glon_t, gclat_t)

xi_p, eta_p = grid.projection.geo2cube(glonmap, gclatmap)

## Set up lompe model object so that we can rip out convection matrices and stuff

Kp   = 4     # for Hardy model
F107 = 100   # for EUV conductance

# functions for conductances to be passed to the Lompe model
SH = lambda lon = grid.lon, lat = grid.lat: lompe.conductance.hardy_EUV(lon, lat, Kp, time, 'hall')
SP = lambda lon = grid.lon, lat = grid.lat: lompe.conductance.hardy_EUV(lon, lat, Kp, time, 'pedersen')

v_e = np.zeros(Npt)
v_n = np.zeros(Npt)

v_tot = np.vstack((v_e.flatten(), v_n.flatten()))

# v_coords = np.vstack((points['glon'].values, points['gclat'].values)) # E-field is defined on the grid
v_coords = np.vstack((glonmap, gclatmap)) # E-field is defined at locations MAPPED from grid to the ionosphere!
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
l1 = 0.00001
l1 = 1e2
l1 = 1e-1
LTL = l1 * np.median(GTG)*np.diag(np.ones(GTG.shape[0]))

if not np.isclose(l1,0.):
    print(f"Regularizing result with l1={l1}")

Cpm = np.linalg.inv(GTG+LTL)

Gobss = model.matrix_func['convection'](lon=grid.lon,lat=grid.lat)
Gobs = np.vstack([G_ for i, G_ in enumerate(Gobss) if i in ds.components])
Cpd = Gobs@Cpm@Gobs.T

vm_e_var = np.diag(Cpd[:Neval,:Neval])   # eastward v-model variance
vm_n_var = np.diag(Cpd[Neval:,Neval:])   # northward v-model variance

doresplot = True
if doresplot:
   from lompe.model.visualization import resolutionplot,locerrorplot
   model.Rmatrix = Cpm.dot(GTG)
   import e3doubt.analysis_utils
   reload(e3doubt.analysis_utils)
   from e3doubt.analysis_utils import calc_resolution
   calc_resolution(model)
   resolutionplot(model)
   locerrorplot(model)

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
        csax.plot(clo, cla, color='black', linewidth=1)
