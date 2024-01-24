#
# A simple comparison of actual errors in GUISDAP fits vs e3doubt estimates. 
#
#
# IV 2023
#

import e3doubt
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import datetime

from geospacelab.datahub import DataHub
import geospacelab.express.eiscat_dashboard as eiscat

ylim = [80,600]
ylim = [100,500]

# time span, site, antenna and modulation for geospacelab. Select one of the experiments below and comment the others
 
# # UHF beata 20210309
# dt_fr = datetime.datetime.strptime('20210309' + '0000', '%Y%m%d%H%M')   # datetime from
# dt_to = datetime.datetime.strptime('20210309' + '0100', '%Y%m%d%H%M')   # datetime to
# site = 'UHF'                # facility attributes required, check from the eiscat schedule page
# antenna = 'UHF'
# modulation = '60'

# # VHF bella 20230815
# dt_fr = datetime.datetime.strptime('20230815' + '1800', '%Y%m%d%H%M')   # datetime from
# dt_to = datetime.datetime.strptime('20230815' + '1900', '%Y%m%d%H%M')   # datetime to
# site = 'VHF'                # facility attributes required, check from the eiscat schedule page
# antenna = 'VHF'
# modulation = '54'

# UHF beata 20230320
dt_fr = datetime.datetime.strptime('20230320' + '1000', '%Y%m%d%H%M')   # datetime from
dt_to = datetime.datetime.strptime('20230320' + '1100', '%Y%m%d%H%M')   # datetime to
site = 'UHF'                # facility attributes required, check from the eiscat schedule page
antenna = 'UHF'
modulation = '60'

# # UHF beata 20080604
# dt_fr = datetime.datetime.strptime('20080604' + '1200', '%Y%m%d%H%M')   # datetime from
# dt_to = datetime.datetime.strptime('20080604' + '1600', '%Y%m%d%H%M')   # datetime to
# site = 'UHF'                # facility attributes required, check from the eiscat schedule page
# antenna = 'UHF'
# modulation = '60'

# # VHF tau8 20081130
# dt_fr = datetime.datetime.strptime('20081130' + '0800', '%Y%m%d%H%M')   # datetime from
# dt_to = datetime.datetime.strptime('20081130' + '1000', '%Y%m%d%H%M')   # datetime to
# site = 'VHF'                # facility attributes required, check from the eiscat schedule page
# antenna = 'VHF'
# modulation = '60'



# we will read EISCAT data from madrigal
database_name = 'madrigal'      # built-in sourced database name 
facility_name = 'eiscat'        # facility name


# create a geospacelab datahub instance
dh = DataHub(dt_fr, dt_to)
# dock the EISCAT data set
ds_isr = dh.dock(datasource_contents=[database_name, 'isr', facility_name],
                      site=site, antenna=antenna, modulation=modulation, data_file_type='eiscat-hdf5')
# load data. Will be downloaded automatically if a local file is not available
ds_isr.load_data()
# assign assign variables...
Ne = dh.assign_variable('n_e')
Te = dh.assign_variable('T_e')
Ti = dh.assign_variable('T_i')
Vi = dh.assign_variable('v_i_los')
Op = dh.assign_variable('comp_O_p')
nu = dh.assign_variable('nu_i')
TXaz = dh.assign_variable('AZ')
TXel = dh.assign_variable('EL')
ts = dh.assign_variable('DATETIME_1')
te = dh.assign_variable('DATETIME_2')
txpow = dh.assign_variable('P_Tx')
hh = dh.assign_variable('HEIGHT')
rr = dh.assign_variable('RANGE')
Tsys = dh.assign_variable('T_SYS_1')
status = dh.assign_variable('STATUS')
chisqr = dh.assign_variable('RESIDUAL')
exper = ds_isr.pulse_code
radar = ds_isr.site
TXlat = ds_isr.metadata['r_XMITloc'][0]
TXlon = ds_isr.metadata['r_XMITloc'][1]
TXalt = ds_isr.metadata['r_XMITloc'][2]
RXlat = ds_isr.metadata['r_RECloc'][0]
RXlon = ds_isr.metadata['r_RECloc'][1]
RXalt = ds_isr.metadata['r_RECloc'][2]

nt = hh.value.shape[0]
nh = hh.value.shape[1]
heights = hh.value


# correct the geospacelab Te errors...
trerr = np.sqrt( (Te.error**2/Te.value**2 - Ti.error**2/Ti.value**2) * Te.value**2/Ti.value**2 )
dte = np.sqrt(Ti.value**2*trerr**2 - Te.value**2/Ti.value**2*Ti.error**2) # error in Te assuming Te and Ti are uncorrelated and Gaussian, https://en.wikipedia.org/wiki/Ratio_distribution#Normal_ratio_distributions
Te.error = dte


# plot the EISCAT data
# NOTE: this is probably repeating things that were already done above, but I would need to learn geospacelab better to figure out how to merge these two..
dashboard = eiscat.EISCATDashboard(
    dt_fr, dt_to,
    site=site, antenna=antenna, modulation=modulation,
    data_file_type='eiscat-hdf5', load_mode='AUTO',
    figure_config={'figsize': (8, 8)})

dashboard.status_mask(bad_status=[1, 2, 3])
dashboard.residual_mask()

n_e = dashboard.assign_variable('n_e')
n_e.visual.axis[1].lim = ylim
n_e.visual.axis[0].data_res = 60
n_e.visual.axis[2].lim = [10**10, 10**12]
n_e.visual.plot_config.pcolormesh.update(cmap='jet')

T_i = dashboard.assign_variable('T_i')
T_i.visual.axis[1].lim = ylim
T_i.visual.axis[2].lim = [0, 2500]
T_i.visual.axis[0].data_res = 60
T_i.visual.plot_config.pcolormesh.update(cmap='jet')

T_e = dashboard.assign_variable('T_e')
T_e.visual.axis[1].lim = ylim
T_e.visual.axis[2].lim = [0, 3000]
T_e.visual.axis[0].data_res = 60
T_e.visual.plot_config.pcolormesh.update(cmap='jet')

v_i = dashboard.assign_variable('v_i_los')
v_i.visual.axis[1].lim = ylim
v_i.visual.axis[0].data_res = 60
v_i.visual.axis[2].lim = [-150, 150]
#v_i.visual.axis[2].lim = [-500, 500]
v_i.visual.plot_config.pcolormesh.update(cmap='jet')

layout = [[n_e], [T_e], [T_i], [v_i]]
dashboard.set_layout(panel_layouts=layout, left=0.12, right=0.87)
# dashboard.draw()
# dashboard.save_figure()
#dashboard.show()

# quartiles and median in GUISDAP data
NeQuant = np.quantile(Ne.value,[.25, .50, .75],axis=0)
TiQuant = np.quantile(Ti.value,[.25, .50, .75],axis=0)
TeQuant = np.quantile(Te.value,[.25, .50, .75],axis=0)
ViQuant = np.quantile(Vi.value,[.25, .50, .75],axis=0)
nuQuant = np.quantile(nu.value,[.25, .50, .75],axis=0)

# # standard deviations
# NeStd = np.std(Ne.value,axis=0)
# TiStd = np.std(Ti.value,axis=0)
# TeStd = np.std(Te.value,axis=0)
# ViStd = np.std(Vi.value,axis=0)


# standard deviations from the inter-quartile range
# this may be more robust than std if there are outliers
NeStd = (NeQuant[2,:]-NeQuant[0,:])/1.34
TiStd = (TiQuant[2,:]-TiQuant[0,:])/1.34
TeStd = (TeQuant[2,:]-TeQuant[0,:])/1.34
ViStd = (ViQuant[2,:]-ViQuant[0,:])/1.34

# median error in guisdap fits
NeErrMed = np.quantile(Ne.error,.5,axis=0)
TiErrMed = np.quantile(Ti.error,.5,axis=0)
TeErrMed = np.quantile(Te.error,.5,axis=0)
ViErrMed = np.quantile(Vi.error,.5,axis=0)


# known properties of the radar systems and experiments
if site=='VHF':
    antGain = 10**4.31 / np.sqrt(2)
    radarFreq = 224e6
    dutyCycle = .125
    TXname = 'TRO'
    RXname = 'TRO'
    if exper=='bella':
        bitlen = 45
    elif exper=='beata':
        bitlen = 20
    elif exper=='tau1':
        bitlen = 72
    elif exper=='tau7':
        bitlen = 120
    elif exper=='tau8':
        bitlen = 84
    elif exper=='manda':
        bitlen = 2.4
    else:
        raise ValueError('Unknown experiment name '+exper+' for site '+site)
elif site=='UHF':
    antGain = 10**4.81
    radarFreq = 930e6
    dutyCycle = .125
    TXname = 'TRO'
    RXname = 'TRO'
    if exper=='bella':
        bitlen = 45
    elif exper=='beata':
        bitlen = 20
    elif exper=='tau1':
        bitlen = 60
    elif exper=='manda':
        bitlen = 2.4
    elif exper=='arc1':
        bitlen = 6
    else:
        raise ValueError('Unknown experiment name '+exper+' for site '+site)
else:
    raise ValueError('Unknown radar site '+site)


print('  ')        
# convert bit lengths to km
fwhmRange = float(bitlen*.299792458/2)
print('Decoded range resolution '+str(fwhmRange))

# approximate beam width from antenna gain
beamwidth = np.sqrt(2/antGain) * 2*np.sqrt(2*np.log(2)) * 180 / np.pi
print('beam width '+str(beamwidth))


# approximative gate widths (are the actual values available somewhere??
dran = np.diff(rr.value[0,])
resR = np.pad(dran,(1,0),'constant',constant_values=(dran[0]))

# date and time resolution
measdate = (ts.value[0,0]+(ts.value[0,0]-ts.value[:,0]).mean())#np.mean(ts.value[:,0])
dt = np.mean(te.value[:,0]-ts.value[:,0])
timeres = float(dt.seconds)
print('Time resolution ' + str(timeres)+' s')

# transmitter power
PTX = float(np.mean(txpow.value[:,0])*1000)
print('TX power '+str(PTX*1e-6)+' MW')

# system noise temperature
Tnoise = float(np.median(Tsys.value[:,0]))
print('Noise temperature '+str(Tnoise)+' K')

print('  ')        

# radar system parameters
RADAR_PARMS = dict(fradar=radarFreq,  # Radar frequency [Hz]
                   dutyCycle=dutyCycle,  # Transmitter duty cycle
                   RXduty=(1,),      # Receiver duty cycle
                   Tnoise=Tnoise, # Noise temperature for receiver sites
                   Pt=PTX,
                   mineleTrans=30,
                   mineleRec=(30,),
                   phArrTrans=False,
                   phArrRec=(False,),
                   fwhmRange=fwhmRange
                   )

    
# initialize the radar experiments
experiment = e3doubt.Experiment(az=TXaz.value[0,0],el=TXel.value[0,0],h=hh.value[0,],refdate_models=measdate,transmitter=TXname,receivers=[RXname],fwhmtx=beamwidth,fwhmrx=[beamwidth],radarparms=RADAR_PARMS)

# set the resolutions etc. (probably could have done this already above?)
experiment.set_range_resolution(resR)
experiment.set_radarparm('fwhmRange',fwhmRange)
experiment.set_radarparm('Pt',float(txpow.value[0,0]*1000))
experiment.set_radarparm('Tnoise',float(Tsys.value[0,0]))
experiment.set_radarparm('fradar',radarFreq)
experiment.set_radarparm('dutyCycle',dutyCycle)

# run IRI and MSIS
experiment.run_models()



# replace model values with median of the measurements
experiment.set_ionos('ne',NeQuant[1,])
experiment.set_ionos('Te',TeQuant[1,])
experiment.set_ionos('Ti',TiQuant[1,])
experiment.set_ionos('nuin',nuQuant[1,])


# Calculate the parameter error estimates
parerrs = experiment.get_uncertainties(integrationsec=timeres,fwhmIonSlab=50)
atmos = experiment.get_atmos()
ionos = experiment.get_ionos()


##############################
#Prepare for plot

#pull data out of dashboard (spent about two hours just tracking this down)
nestuff = dashboard.panels[0]._retrieve_data_2d(dashboard.panel_layouts[0][0])
testuff = dashboard.panels[0]._retrieve_data_2d(dashboard.panel_layouts[1][0])
tistuff = dashboard.panels[0]._retrieve_data_2d(dashboard.panel_layouts[2][0])
vilosstuff = dashboard.panels[0]._retrieve_data_2d(dashboard.panel_layouts[3][0])


getem = lambda you: (you['x'], you['y'], you['z'])

def fixemup(x,y,z):
    if x.shape[0] == z.shape[0]:
        delta_x = np.diff(x, axis=0)
        x[:-1, :] = x[:-1, :] + delta_x/2
        x = np.vstack((
            np.array(x[0, 0] - delta_x[0, 0] / 2).reshape((1, 1)),
            x[:-1, :],
            np.array(x[-1, 0] + delta_x[-1, 0] / 2).reshape((1, 1))
        ))
    if len(y.shape) == 1:
        y = y[np.newaxis, :]
    if y.shape[1] == z.shape[1]:
        delta_y = np.diff(y, axis=1)
        y[:, :-1] = y[:, :-1] + delta_y/2
        y = np.hstack((
            np.array(y[:, 0] - delta_y[:, 0]/2).reshape((y.shape[0], 1)),
            y[:, :-1],
            np.array(y[:, -1] + delta_y[:, -1]/2).reshape((y.shape[0], 1)),
        ))

    if y.shape[0] == z.shape[0]:
        y = np.vstack((y, y[-1, :].reshape((1, y.shape[1]))))

    return (x,y,z)

def get_znorm(z,z_lim=None):
    if z_lim is None:
        z_lim = [np.nanmin(z.flatten()), np.nanmax(z.flatten())]
    norm = mpl_colors.LogNorm(vmin=z_lim[0], vmax=z_lim[1])
    return norm


##############################
# NOw the plot

fontsize = 13
plt.rcParams['font.size'] = str(fontsize)
params = {'axes.labelsize': fontsize,'axes.titlesize':fontsize, 'legend.fontsize': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
plt.rcParams.update(params)

nrows,ncols = 28,32

fig = plt.figure(3,figsize=(12,12))
plt.clf()

gs = fig.add_gridspec(nrows,ncols)

margin = 0.065
plt.subplots_adjust(left=0.08,right=0.9,bottom=margin,top=0.96,hspace=1.5,wspace=2.0)

cols_cax = 2

rows_rowpanel = 4
cols_rowpanel = ncols-cols_cax+1


rowpanel_colpanel_rowstwixt = 1

rows_colpanel = 10
cols_colpanel = ncols//4

axne = fig.add_subplot(gs[0*rows_rowpanel:1*rows_rowpanel, :cols_rowpanel])
axte = fig.add_subplot(gs[1*rows_rowpanel:2*rows_rowpanel, :cols_rowpanel])
axti = fig.add_subplot(gs[2*rows_rowpanel:3*rows_rowpanel, :cols_rowpanel])
axvi = fig.add_subplot(gs[3*rows_rowpanel:4*rows_rowpanel, :cols_rowpanel])

caxne = fig.add_subplot(gs[0*rows_rowpanel:1*rows_rowpanel, cols_rowpanel:])
caxte = fig.add_subplot(gs[1*rows_rowpanel:2*rows_rowpanel, cols_rowpanel:])
caxti = fig.add_subplot(gs[2*rows_rowpanel:3*rows_rowpanel, cols_rowpanel:])
caxvi = fig.add_subplot(gs[3*rows_rowpanel:4*rows_rowpanel, cols_rowpanel:])

axdne = fig.add_subplot(gs[4*rows_rowpanel+rowpanel_colpanel_rowstwixt:,
                           0*cols_colpanel:1*cols_colpanel])
axdte = fig.add_subplot(gs[4*rows_rowpanel+rowpanel_colpanel_rowstwixt:,
                           1*cols_colpanel:2*cols_colpanel])
axdti = fig.add_subplot(gs[4*rows_rowpanel+rowpanel_colpanel_rowstwixt:,
                           2*cols_colpanel:3*cols_colpanel])
axdvi = fig.add_subplot(gs[4*rows_rowpanel+rowpanel_colpanel_rowstwixt:,
                           3*cols_colpanel:4*cols_colpanel])

fig.suptitle(radar+" "+exper+" "+dt_fr.strftime("%Y/%m/%d %H:%M–")+dt_to.strftime("%H:%M UT")+r" (T$_{sys}$ = "+f"{Tsys.value[0,0]:.0f} K)",fontsize=fontsize+2)


cmap = 'viridis'
import matplotlib.colors as mpl_colors
import matplotlib.dates as mdates

x,y,z = fixemup(*getem(nestuff))
zlim = [1e10,1e12]
norm = get_znorm(z,z_lim=zlim)
imne = axne.pcolormesh(x.T,y.T,z.T,norm=norm,cmap=cmap)

# te
x,y,z = fixemup(*getem(testuff))
zlim = [np.nanmin(z.flatten()), np.nanmax(z.flatten())]
zlim = [0,3000]
imte = axte.pcolormesh(x.T,y.T,z.T,cmap=cmap,vmin=zlim[0],vmax=zlim[1])

# ti
x,y,z = fixemup(*getem(tistuff))
zlim = [np.nanmin(z.flatten()), np.nanmax(z.flatten())]
zlim = [0,2500]
imti = axti.pcolormesh(x.T,y.T,z.T,cmap=cmap,vmin=zlim[0],vmax=zlim[1])

# vi
x,y,z = fixemup(*getem(vilosstuff))
zlim = [np.nanmin(z.flatten()), np.nanmax(z.flatten())]
vminmax = 150
imvi = axvi.pcolormesh(x.T,y.T,z.T,cmap='bwr',vmin=-vminmax,vmax=vminmax)

cbne = fig.colorbar(imne,cax=caxne)
cbte = fig.colorbar(imte,cax=caxte)
cbti = fig.colorbar(imti,cax=caxti)
cbvi = fig.colorbar(imvi,cax=caxvi)

cbne.set_label(r"$n_e$ [m$^{-3}$]")
cbte.set_label(r"$T_e$ [K]")
cbti.set_label(r"$T_i$ [K]")
cbvi.set_label(r"$v_i$ [m/s]")

ylab_colpanel = 'Alt [km]'
colpanels = [axne,axte,axti,axvi]
count = 1
for colpanel in colpanels:
    colpanel.set_ylim(ylim)

    colpanel.set_ylabel(ylab_colpanel)
    if count < len(colpanels):
        colpanel.get_xaxis().set_ticklabels([])
    else:
        colpanel.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
    count += 1
    

stylegui = '-'
stylee3d = ':'
stylesample = '-'

cgui = 'lightgray'
ce3d = 'black'
csample = 'orange'

lwgui = 3
lwe3d = 2
lwsample = 1

ne = ionos.loc[:,'ne']
errne = parerrs.loc[:,'dne1']
axdne.plot(NeErrMed,hh.value[0,:],ls=stylegui,color=cgui,label='GUISDAP',lw=lwgui)
axdne.plot(errne,hh.value[0,:],ls=stylee3d,color=ce3d,label='e3doubt',lw=lwe3d)
axdne.plot(NeStd,hh.value[0,:],ls=stylesample,color=csample,label='sample',lw=lwsample)
axdne.set_xlim([0,5e10])
axdne.set_ylim(ylim)
axdne.set_ylabel('Alt [km]')
axdne.set_xlabel('std($n_e$) [m$^{-3}$]')
l1 = axdne.legend()

axdte.plot(TeErrMed,hh.value[0,:],ls=stylegui,color=cgui,label='GUISDAP',lw=lwgui)
axdte.plot(parerrs.loc[:,'dTe1'],hh.value[0,:],ls=stylee3d,color=ce3d,label='e3doubt',lw=lwe3d)
axdte.plot(TeStd,hh.value[0,:],ls=stylesample,color=csample,label='sample',lw=lwsample)
axdte.set_xlim([0,330])
axdte.set_ylim(ylim)
axdte.set_xlabel('std($T_e$) [K]')
# l2 = axdte.legend()

axdti.plot(TiErrMed,hh.value[0,:],ls=stylegui,color=cgui,label='GUISDAP',lw=lwgui)
axdti.plot(parerrs.loc[:,'dTi1'],hh.value[0,:],ls=stylee3d,color=ce3d,label='e3doubt',lw=lwe3d)
axdti.plot(TiStd,hh.value[0,:],ls=stylesample,color=csample,label='sample',lw=lwsample)
axdti.set_xlim([0,330])
axdti.set_ylim(ylim)
axdti.set_xlabel('std($T_i$) [K]')
# lti = axdti.legend()


axdvi.plot(ViErrMed,hh.value[0,:],ls=stylegui,color=cgui,label='$dV_i (GUISDAP)$',lw=lwgui)
axdvi.plot(parerrs.loc[:,'dVi1'],hh.value[0,:],ls=stylee3d,color=ce3d,label='$dV_i (e3doubt)$',lw=lwe3d)
axdvi.plot(ViStd,hh.value[0,:],ls=stylesample,color=csample,label='$dV_i (sample)$',lw=lwsample)
axdvi.set_xlim([0,90])
axdvi.set_ylim(ylim)
axdvi.set_xlabel('std($v_i$) [m/s]')
# l3 = axdvi.legend()

rowpanels = [axdne,axdte,axdti,axdvi]
count = 1
for panel in rowpanels:
    panel.set_ylim(ylim)

    _ = panel.spines['right'].set_visible(False)
    _ = panel.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    _ = panel.yaxis.set_ticks_position('left')
    _ = panel.xaxis.set_ticks_position('bottom')

    if count > 1:
        panel.get_yaxis().set_ticklabels([])
    count += 1


# label all panels with a letter
labs = ['a','b','c','d','e','f','g','h']
pancount = 0
for pan in colpanels:
    pan.text(0.02, 0.85, labs[pancount],
            fontsize='xx-large',
            fontweight='bold',
            horizontalalignment='center',
            verticalalignment='center',
            transform = pan.transAxes)
    pancount += 1

for pan in rowpanels:
    pan.text(0.07, 0.95, labs[pancount],
            fontsize='xx-large',
            fontweight='bold',
            horizontalalignment='center',
            verticalalignment='center',
            transform = pan.transAxes)
    pancount += 1
    
plt.savefig("error_comparison_"+radar+"_"+exper+"_"+measdate.strftime("%Y%m%dT%H%M%S")+".png",dpi=300)
# plt.show()


