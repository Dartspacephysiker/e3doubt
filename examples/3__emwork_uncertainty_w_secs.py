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

import e3doubt.utils
reload(e3doubt.utils)
from e3doubt.utils import get_perpendicular_velocity_mapping_matrix

from apexpy import Apex

refdate_models = datetime(2020,12,1)

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
Lgrid = 80e3
Wgrid = 80e3
Lresgrid = Wresgrid = 10e3 #grid resolution in m                                 
wshiftgrid = -Wresgrid//2
# wshiftgrid = 0.
az, el, gdlatg, glong, h, gridm = get_2D_csgrid_az_el(gdlat_t, glon_t, h_grid=h_grid,
                                                      L=Lgrid, W=Wgrid,
                                                      Lres=Lresgrid, Wres=Wresgrid,
                                                      wshift=wshiftgrid,
                                                      return_grid=True)

H  = np.array([200,220])        # Altitude(s) from which to estimate convection 
H_vi = np.array([100])          # Altitude(s) at which to calculate EM work = J dot E

# convection estimation experiment
exp = experiment.Experiment(az=az,el=el,h=H,refdate_models=refdate_models)
exp.run_models()
dfunc = exp.calc_uncertainties(integrationsec=300)

# EM work estimation experiment
expvi = experiment.Experiment(az=az,el=el,h=H_vi,refdate_models=refdate_models)
expvi.run_models()
dfuncvi = expvi.calc_uncertainties(integrationsec=300)


# get coordinates of convection estimation points
points = exp.get_points()
radarconfig = exp.get_radarconfig()
Npt = exp.N['pt']

apex_refh = 110
mapto_h = apex_refh

gdlat, glon, h = points['gdlat'].values,points['glon'].values, points['h'].values

# Get matrix that lets us map covariance to ionospheric altitude
B, (gdlatmap, glonmap) = get_perpendicular_velocity_mapping_matrix(gdlat,glon,h,
                                                                   apex_refh=apex_refh,
                                                                   mapto_h=mapto_h,
                                                                   refdate=refdate_models,
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
Lres = Wres = 10e3
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
v_coords = np.vstack((glonmap, gclatmap)) # E-field is measured at MAPPED locations the grid
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
l1 = 1e-3
# l1 = 1e2
l1 = 1e-1
LTL = l1 * np.median(GTG)*np.diag(np.ones(GTG.shape[0]))

if not np.isclose(l1,0.):
    print(f"Regularizing result with l1={l1}")

Cpm = np.linalg.inv(GTG+LTL)

# Gobss = model.matrix_func['convection'](lon=grid.lon,lat=grid.lat)
Gobss = model.matrix_func['efield'](lon=grid.lon,lat=grid.lat)
Gobs = np.vstack([G_ for i, G_ in enumerate(Gobss) if i in ds.components])
Cpd = Gobs@Cpm@Gobs.T

Em_e_var = np.diag(Cpd[:Neval,:Neval])   # eastward E-model variance, (V/m)²
Em_n_var = np.diag(Cpd[Neval:,Neval:])   # northward E-model variance

# Convert to (mV/m)²
Em_e_var = Em_e_var * 1e6
Em_n_var = Em_n_var * 1e6


doresplot = False
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
                  color='C1',
                  label='obs ('+", ".join([str(h) for h in H])+" km)")

csaxes[0].scatter(glonmap, gclatmap,
                  marker='v', s=10,
                  zorder=21,
                  alpha=0.5,
                  color='C2',
                  label='mapped ('+str(mapto_h)+' km)')

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

axes[2].set_title("Model $E_e$ variance @ "+str(mapto_h)+"-km alt")
axes[3].set_title("Model $E_n$ variance @ "+str(mapto_h)+"-km alt")

# cmap = plt.get_cmap('bwr')
cmap = plt.get_cmap('bwr').resampled(10)
cvme = csaxes[2].pcolormesh(grid.lon, grid.lat, Em_e_var.reshape(grid.lon.shape),
                            cmap=cmap,vmin=pcmeshlevels[0],vmax=pcmeshlevels[-1])
cvmn = csaxes[3].pcolormesh(grid.lon, grid.lat, Em_n_var.reshape(grid.lon.shape),
                          cmap=cmap,vmin=pcmeshlevels[0],vmax=pcmeshlevels[-1])


for csax in csaxes:
    csax.ax.set_xlim((-0.02,0.02))
    csax.ax.set_ylim((-0.02,0.02))

cbe = plt.colorbar(cvme) 
cbn = plt.colorbar(cvmn)

cbe.set_label("(mV/m)²")
cbn.set_label("(mV/m)²")

glonb, gclatb = gridedges(gridm.lon,gridm.lat)
for csax in csaxes[2:]:
    csax.plot(glonb,gclatb,color='k',alpha=0.5)


HIGHRES = False
for cl in grid.projection.get_projected_coastlines(resolution='10m' if HIGHRES else '50m'):
    clo, cla = grid.projection.cube2geo(cl[0], cl[1])

    xis, etas = gridm.projection.geo2cube(clo, cla) 
    for csax in csaxes:
        csax.plot(clo, cla, color='black', linewidth=1)


##############################
# Now get E-field covariance matrices at observation points

lonvi, latvi = expvi.get_points()['glon'].values,expvi.get_points()['gclat'].values

Gvis = model.matrix_func['efield'](lon=lonvi,lat=latvi)

print("TECHNICALLY you should map the E-field covariances to the appropriate altitude!")
Gvi = np.vstack([G_ for i, G_ in enumerate(Gvis) if i in ds.components])
Cpdvi = Gvi@Cpm@Gvi.T

Nvieval = expvi.N['pt']
Emvi_e_var = np.diag(Cpdvi[:Nvieval,:Nvieval])   # eastward E-model variance, (V/m)²
Emvi_n_var = np.diag(Cpdvi[Nvieval:,Nvieval:])   # northward E-model variance

# off-diagonal terms
Emvi_en_var = np.diag(Cpdvi[:Nvieval,Nvieval:])   # eastward E-model variance, (V/m)²
Emvi_ne_var = np.diag(Cpdvi[Nvieval:,:Nvieval])   # northward E-model variance

assert np.all(np.isclose(Emvi_ne_var,Emvi_en_var))  # Sanity check, off-diag elements should be the same regardless of echelon

# Make covariance matrices for each point
covE = np.vstack([Emvi_e_var,Emvi_en_var,Emvi_ne_var,Emvi_n_var]).reshape(2,2,expvi.N['pt'])

# MAP E-field covariance matrix to relevant altitude
print("WARNING: Right now you do not have code for mapping the E-field covariance to a different altitude! It's technically (and possibly seriously, depending on altitude in question) incorrect to not map E-fields to the height of the observations")
# Get B matrix for E-fields by writing code to do it

print("WARNING: You also have no calculation of the upward component of E and elements of E's covariance matrix involving the upward component")
# Figure this out


## Get ion velocity covariance at measurement altitude
cov_vi = expvi.get_velocity_cov_matrix()

points_vi = expvi.get_points()

gdlat_vi, glon_vi, h_vi = points_vi['gdlat'].values,points_vi['glon'].values, points_vi['h'].values

# Map all these to ONE height
# The goal is actually not mapping, but to get rid of the covariance of the field-aligned component of the ion drift velocity variance
assert np.all(np.isclose(h_vi[0],h_vi)), "In what I do in what follows I assume that there is only one height to map to!"
B_vi, (gdlatmap_vi, glonmap_vi) = get_perpendicular_velocity_mapping_matrix(gdlat_vi,glon_vi,h_vi,
                                                                            apex_refh=apex_refh,
                                                                            mapto_h=h_vi[0],
                                                                            refdate=refdate_models,
                                                                            return_mapped_coordinates=True)

cov_viperp = np.transpose(np.einsum('ij...,jk...,lk...',B_vi,cov_vi,B_vi),axes=[1,2,0])  # use 'lk' and not 'kl' at the end because 

##Next, combine E-field and density (co)variances with ion velocity variances
uncvi = expvi.calc_uncertainties(integrationsec=300)

var_n = uncvi['dnemulti'].values.squeeze()**2

covar_vi = cov_viperp
covar_e = covE

n = expvi.get_ionos('ne').values.squeeze()

SEED = 2020
rng = np.random.default_rng(SEED)

vimag = 50
vimag = 400

Emag = vimag*5e4*1e-9

vimean = np.array([vimag/np.sqrt(2),vimag/np.sqrt(2)])
Emean = np.array([0,Emag])          # in V/m


vi = rng.normal(loc=vimean[:,np.newaxis],
                scale=np.sqrt(np.stack([covar_vi[0,0],covar_vi[1,1]])),
                size=(2,n.size))  # scale (stddev) in m/s

E = rng.normal(loc=Emean[:,np.newaxis],
               scale=np.sqrt(np.stack([covar_e[0,0],covar_e[1,1]])),
               size=(2,n.size))     # scale (stddev) in V/m, assuming typical E-field is order 10s of mV/m

vi_zero = rng.normal(loc=np.array([0,0])[:,np.newaxis],
                scale=np.sqrt(np.stack([covar_vi[0,0],covar_vi[1,1]])),
                size=(2,n.size))  # scale (stddev) in m/s
E_zero = rng.normal(loc=np.array([0,0])[:,np.newaxis],
               scale=np.sqrt(np.stack([covar_e[0,0],covar_e[1,1]])),
               size=(2,n.size))     # scale (stddev) in V/m, assuming typical E-field is order 10s of mV/m


q = 1.602176634e-19         # elementary charge
emwork = q * n * np.sum(vi * E,axis=0)

emwork_zero = q * n * np.sum(vi_zero * E_zero,axis=0)

def var_emwork(n, vi, E, var_n, covar_vi, covar_e,hack=True):
    """
    The dimension corresponding to measurement number should be the last dimension of vi, E, covar_vi, covar_e
    """

    cv, ce = covar_vi, covar_e
    
    nsq = n**2
    
    if hack:
        print("var_emwork HACK: Eliminating contribution to covariance from upward component since you don't currently do anything with the upward component")
        
        vi = vi[:2]
        E = E[:2]
    
        cv = cv[:2,:2]
        ce = ce[:2,:2]
        
    #handle each term separately
    
    # Represents E(vi)^T Cov_E E(vi), where E() is expectation value
    evcove = np.einsum("i...,ij...,j...",vi, ce, vi)
    
    # Represents E(E)^T Cov_vi E(E)
    eecovv = np.einsum("i...,ij...,j...",E, cv, E)
    
    # Represents Trace(Cov_v Cov_E)
    trcvce = np.einsum("ij...,ij...",cv,ce)

    t1 = nsq * evcove
        
    t2 = nsq * eecovv
    
    t3 = nsq * trcvce
    
    t4 = var_n * np.sum(vi*vi,axis=0) * np.sum(E*E,axis=0)
    #t4alt = var_n * np.einsum("i...,i...",vi,vi) * np.einsum("i...,i...",E,E)
    
    t5 = var_n * eecovv
    
    t6 = var_n * evcove
    
    t7 = var_n * trcvce
    
    est1 = q**2 * (t1+t2+t3+t4+t5+t6+t7)
    
    # This expression apparently incurs rounding error. Strange!
    # est2 = nsq * (evcove + eecovv + trcvce) \
    #     + var_n * (np.sum(vi*vi,axis=0) * np.sum(E*E,axis=0) + eecovv + evcove + trcvce)
    
    # return est1,est2
    return est1

varem = var_emwork(n, vi, E, var_n, covar_vi, covar_e)

varem_zero = var_emwork(n, vi_zero, E_zero, var_n, covar_vi, covar_e)

import matplotlib.pyplot as plt
plt.ion()

scale = 1e-9

bins = np.linspace(0,np.abs(emwork).max()/scale*10,150)

fig = plt.figure(num=1,figsize=(8,6))

plt.clf()

fig.suptitle(r"$\lambda_1$ = "+f"{l1:.5e}\n"+
             "$\mathbf{v}_i$ = ["+", ".join([f"{v:.1f}" for v in vimean])+"] m/s, "+
             "$\mathbf{E}$ = ["+", ".join([f"{v*1e3:.1f}" for v in Emean])+"] mV/m")

# ax0 = plt.subplot(1,2,1)
# ax1 = plt.subplot(1,2,2)
ax0 = plt.subplot(1,1,1)

plt.sca(ax0)

# plt.hist(np.abs(emwork)/scale,label="emwork",alpha=0.5)#,bins=bins)
# plt.hist(np.abs(np.sqrt(varem))/scale,label="stddev(em)",alpha=0.5)#,bins=bins)
# plt.hist(np.abs(np.sqrt(varem_zero))/scale,label="stddev(em) (E=vi=0)",alpha=0.5)#,bins=bins)

# shows = [np.abs(emwork)/scale,
#          np.abs(np.sqrt(varem))/scale,
#          np.abs(np.sqrt(varem_zero))/scale]

# labs = ['abs(emwork)','stddev(em)','stddev(em) (E=vi=0)']
# plt.hist(shows,label=labs)

emworkstr = "$\mathbf{j}\cdot\mathbf{E}$"
plt.hist(np.abs(emwork)/scale,label=emworkstr+" = $q n \mathbf{v}_i \cdot \mathbf{E}$",alpha=0.5)#,bins=bins)
plt.hist(np.abs(np.sqrt(varem))/scale,label="stddev("+emworkstr+")",alpha=0.5)#,bins=bins)
plt.hist(np.abs(np.sqrt(varem_zero))/scale,label="stddev("+emworkstr+") (assume E=vi=0)",alpha=0.5)#,bins=bins)

plt.xlabel("EM work [nW/m³]")

plt.legend()

# plt.sca(ax1)

# plt.hist(np.abs(emwork_zero)/scale,label="emwork_zero",bins=bins)
# plt.legend()

