#
# A simple test of Ne measurement accuracy with different numbers of subarrays in E3D.
#
#
# IV 2026
#

import e3doubt
import numpy as np
import datetime
import sys

# number of subarrays at each site
n_subarrays_SKI_TX = 19
n_subarrays_SKI_RX = 19
n_subarrays_KRS = 19
n_subarrays_KAI = 19

# Transmit power per panel
TX_power_per_subarray = 91*2*500

# range resolution from modulation (bit length) in km
# should not be larger than 15 km, because sampling is assumed to be matched to the bit length
# (we are not prepared for modeling long pulses etc...)
bitLength = 3.

# transmitter beam azimuth array
#TXaz = np.array([0])

# transmitter beam elevation array
#TXel = np.array([90])

# transmitter beam azimuth array
TXaz = np.array([0, 90, 180, 270, 0])

# transmitter beam elevation array
TXel = np.array([45, 45, 45, 45, 90])

# Altitudes, where the errors are estimated
hh = np.array([100., 150., 200.,300., 500., 600.])

# final integrated range resolution at each alttiude
# this can be larger than the bit length. Again, not prepared for long pulses, but
# ~50 km might resemble them rather well.
# Notice: equal lengths of TXax, TXel, hh, and resR will cause an error. Add one extra height to avoid
# this, if needed. 
resR = np.array([3., 3., 6., 12., 24., 36.])
#resR = 3.

# time resolution (total time for all beams, divided equally between the beams)
resT = 60.

# reference data from IRI model
# at least my iri2016 works only until 2020... 
refdate_models = datetime.datetime(2020,6,26,11,0,0)

# radar carrier frequency
radarFreq  = 233e6

# area of one subarray
A_subarray = 42.422

# beam widths based on aperture sizes, assuming circular apertures...
beamwidth_SKI_TX = 70. * 3e8/radarFreq / np.sqrt(4/np.pi * n_subarrays_SKI_TX * A_subarray)
beamwidth_SKI_RX = 70. * 3e8/radarFreq / np.sqrt(4/np.pi * n_subarrays_SKI_RX * A_subarray )
beamwidth_KRS = 70. * 3e8/radarFreq / np.sqrt(4/np.pi * n_subarrays_KRS * A_subarray)
beamwidth_KAI = 70. * 3e8/radarFreq / np.sqrt(4/np.pi * n_subarrays_KAI * A_subarray)


# transmit power
PTX = n_subarrays_SKI_TX * TX_power_per_subarray

# transmitter location in geodetic coordinates
TXloc = ('SKI',69.34,20.31)

# receiver locations
RXloc = [('SKI',69.34,20.31) , ('KRS',68.48,22.52), ('KAI',68.27,19.45)]

# transmitter duty cycle
dutyCycle = 0.25

# "Receiver duty cycle" (typically 1, but could be reduced on the core site if multipurpose codes are used)
rxduty = [1., 1., 1.]

# system noise temperature. Notice that sky-noise increases a lot when moving to lower frequencies!
Tnoise = 250.

# elevation limits for transmission and reception
mineleTrans = 30.
mineleRec = [30., 30., 30.]

# are the antennas phased-arrays? (beams widen when steered away from zenith)
phArrTrans = True
phArrRec = (True,True,True)


# a "layer thickness" to make approximate self-noise calcultions. 50 km has worked well.
NeThickness = 50.

# radar system parameters
RADAR_PARMS = dict(fradar=radarFreq,
                   dutyCycle=dutyCycle,
                   RXduty=rxduty,
                   Tnoise=Tnoise,
                   Pt=PTX,
                   mineleTrans=mineleTrans,
                   mineleRec=mineleRec,
                   phArrTrans=phArrTrans,
                   phArrRec=phArrRec,
                   fwhmRange=bitLength
                   )

    
# initialize the radar experiments
experiment = e3doubt.Experiment(az=TXaz,el=TXel,h=hh,refdate_models=refdate_models,transmitter=TXloc,receivers=RXloc,fwhmtx=beamwidth_SKI_TX,fwhmrx=[beamwidth_SKI_RX,beamwidth_KRS,beamwidth_KAI],radarparms=RADAR_PARMS,resR=resR)

# run IRI and MSIS
experiment.run_models()

# Calculate the parameter error estimates
# Set hTeTi=1000 to force a 3-parameter fit of Ne, Ti, and Vi. The error of Ne is very close
# to that of the raw density (power profile) in this case.
hTeTi = 110
parerrsNe = experiment.get_uncertainties(integrationsec=resT,fwhmIonSlab=NeThickness,hTeTi=1000)
parerrsAll = experiment.get_uncertainties(integrationsec=resT,fwhmIonSlab=NeThickness,hTeTi=hTeTi)

# error estimates of individual Vi components
ViCov = experiment.get_velocity_cov_matrix()
ViStd = np.array([[0.]*3]*TXaz.size*hh.size)
for ii in range(TXaz.size*hh.size):
    cmat = ViCov[:,:,ii]
    for ll in range(3):
        ViStd[ii,ll] = np.sqrt(cmat[ll,ll])
    # replace insanely large values with nans
    fillnan = False
    for ll in range(3):
        if ViStd[ii,ll] > 1.e6:
            fillnan = True
    if fillnan:
        for ll in range(3):
            ViStd[ii,ll] = np.nan
            


# coordinates of the measurement volumes
points = experiment.get_points()

plasma_params = experiment.get_ionos()

# print the results
print('')
print('')
print('')
print("###############################################################################################")

print(f"Number of TX subarrays: {n_subarrays_SKI_TX:3.0f}") 
print(f"Number of RX subarrays in SKI: {n_subarrays_SKI_RX:3.0f}") 
print(f"Number of RX subarrays in KRS: {n_subarrays_KRS:3.0f}") 
print(f"Number of RX subarrays in KAI: {n_subarrays_KAI:3.0f}") 
print(f"Integration time: {resT:.1f} s (total time for all beams, shared equally)")
print(f"Modulation bit length: {bitLength:.1f} km")
print(f"IRI model time: {refdate_models}")
print("")
print("")
print("###############################################################################################")
print("")
print("")
print("Raw electron density at individual sites")
print("")
print(f"TX azim TX elev h (km) resR (km)  Ne (1e11 m^-3)  dNe/Ne (SKI)  dNe/Ne (KRS)  dNe/Ne (KAI)")
for ii in range(TXaz.size*hh.size):
    print( f"{points.loc[ii:ii,'az'].values[0]:7.1f} {points.loc[ii:ii,'el'].values[0]:7.1f} {points.loc[ii:ii,'h'].values[0]:6.0f} {points.loc[ii:ii,'resR'].values[0]:9.1f} {(plasma_params.loc[ii:ii,'ne'].values[0]/1e11):15.3f} {parerrsNe.loc[ii:ii,'dne1'].values[0]/plasma_params.loc[ii:ii,'ne'].values[0]:13.3f} {parerrsNe.loc[ii:ii,'dne2'].values[0]/plasma_params.loc[ii:ii,'ne'].values[0]:13.3f} {parerrsNe.loc[ii:ii,'dne3'].values[0]/plasma_params.loc[ii:ii,'ne'].values[0]:13.3f}" )

print('')
print('')
print('')

print("###############################################################################################")
print("")
print(f"Ne, Te, Ti, Vi at individual sites. Te=Ti below {hTeTi:4.0f} km.")
print("")
print("")
print("Skibotn")
print("")
print(f"TX azim TX elev h (km) resR (km)  Ne (1e11 m^-3) Te (K)  Ti (K) Vi (m/s)     dNe     dTe     dTi     dVi")
for ii in range(TXaz.size*hh.size):
    print( f"{points.loc[ii:ii,'az'].values[0]:7.1f} {points.loc[ii:ii,'el'].values[0]:7.1f} {points.loc[ii:ii,'h'].values[0]:6.0f} {points.loc[ii:ii,'resR'].values[0]:9.1f} {(plasma_params.loc[ii:ii,'ne'].values[0]/1e11):15.2f} {plasma_params.loc[ii:ii,'Te'].values[0]:6.0f}  {plasma_params.loc[ii:ii,'Ti'].values[0]:6.0f}  {0:7.0f} {parerrsAll.loc[ii:ii,'dne1'].values[0]/1e11:7.2f} {parerrsAll.loc[ii:ii,'dTe1'].values[0]:7.0f} {parerrsAll.loc[ii:ii,'dTi1'].values[0]:7.0f} {parerrsAll.loc[ii:ii,'dVi1'].values[0]:7.0f}" )

print('')
print('')
print('')
print("")
print("Karesuvanto")
print("")
print(f"TX azim TX elev h (km) resR (km)  Ne (1e11 m^-3) Te (K)  Ti (K) Vi (m/s)     dNe     dTe     dTi     dVi")
for ii in range(TXaz.size*hh.size):
    print( f"{points.loc[ii:ii,'az'].values[0]:7.1f} {points.loc[ii:ii,'el'].values[0]:7.1f} {points.loc[ii:ii,'h'].values[0]:6.0f} {points.loc[ii:ii,'resR'].values[0]:9.1f} {(plasma_params.loc[ii:ii,'ne'].values[0]/1e11):15.2f} {plasma_params.loc[ii:ii,'Te'].values[0]:6.0f}  {plasma_params.loc[ii:ii,'Ti'].values[0]:6.0f}  {0:7.0f} {parerrsAll.loc[ii:ii,'dne2'].values[0]/1e11:7.2f} {parerrsAll.loc[ii:ii,'dTe2'].values[0]:7.0f} {parerrsAll.loc[ii:ii,'dTi2'].values[0]:7.0f} {parerrsAll.loc[ii:ii,'dVi2'].values[0]:7.0f}" )

print('')
print('')
print('')


print("Kaiseniemi")
print("")
print(f"TX azim TX elev h (km) resR (km)  Ne (1e11 m^-3) Te (K)  Ti (K) Vi (m/s)     dNe     dTe     dTi     dVi")
for ii in range(TXaz.size*hh.size):
    print( f"{points.loc[ii:ii,'az'].values[0]:7.1f} {points.loc[ii:ii,'el'].values[0]:7.1f} {points.loc[ii:ii,'h'].values[0]:6.0f} {points.loc[ii:ii,'resR'].values[0]:9.1f} {(plasma_params.loc[ii:ii,'ne'].values[0]/1e11):15.2f} {plasma_params.loc[ii:ii,'Te'].values[0]:6.0f}  {plasma_params.loc[ii:ii,'Ti'].values[0]:6.0f}  {0:7.0f} {parerrsAll.loc[ii:ii,'dne3'].values[0]/1e11:7.2f} {parerrsAll.loc[ii:ii,'dTe3'].values[0]:7.0f} {parerrsAll.loc[ii:ii,'dTi3'].values[0]:7.0f} {parerrsAll.loc[ii:ii,'dVi3'].values[0]:7.0f}" )

print('')
print('')
print('')
print("###############################################################################################")

print('')
print('')


print(f"Multistatic analysis. Te=Ti below {hTeTi:4.0f} km. Vi components are in geodetic East-North-Up system.")
print("")
print(f"TX azim TX elev h (km) resR (km)  Ne (1e11 m^-3) Te (K)  Ti (K) Vi (m/s)     dNe     dTe     dTi   dVi_E   dVi_N   dVi_U")
for ii in range(TXaz.size*hh.size):
    print( f"{points.loc[ii:ii,'az'].values[0]:7.1f} {points.loc[ii:ii,'el'].values[0]:7.1f} {points.loc[ii:ii,'h'].values[0]:6.0f} {points.loc[ii:ii,'resR'].values[0]:9.1f} {(plasma_params.loc[ii:ii,'ne'].values[0]/1e11):15.2f} {plasma_params.loc[ii:ii,'Te'].values[0]:6.0f}  {plasma_params.loc[ii:ii,'Ti'].values[0]:6.0f}  {0:7.0f} {parerrsAll.loc[ii:ii,'dnemulti'].values[0]/1e11:7.2f} {parerrsAll.loc[ii:ii,'dTemulti'].values[0]:7.0f} {parerrsAll.loc[ii:ii,'dTimulti'].values[0]:7.0f} {ViStd[ii,0]:7.0f} {ViStd[ii,1]:7.0f} {ViStd[ii,2]:7.0f}" )

print('')
print('')
print('')
