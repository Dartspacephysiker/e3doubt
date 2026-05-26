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
n_subarrays_SKI_TX = 1
n_subarrays_SKI_RX = 1
n_subarrays_KRS = 1
n_subarrays_KAI = 1

# Transmit power per panel
TX_power_per_subarray = 1e5

# range resolution from modulation (bit length) in km
# should not be larger than 15 km, because sampling is assumed to be matched to the bit length
# (we are not prepared for modeling long pulses etc...)
bitLength = 15.

# transmitter beam azimuth array
TXaz = np.array([0])

# transmitter beam elevation array
TXel = np.array([90])

# Altitudes, where the errors are estimated
hh = np.array([100., 200.,300.])

# final integrated range resolution at each alttiude
# this can be larger than the bit length. Again, not prepared for long pulses, but
# ~50 km might resemble them rather well. 
resR = 45.

# time resolution
resT = 60.

# reference data from IRI model
# at least my iri2016 works only until 2020... 
refdate_models = datetime.datetime(2020,5,26,10,0,0)

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
parerrs = experiment.get_uncertainties(integrationsec=resT,fwhmIonSlab=NeThickness,hTeTi=1000)

# coordinates of the measurement volumes
points = experiment.get_points()

plasma_params = experiment.get_ionos()

# print the results
print('')
print('')
print('')

print(f"Integration time: {resT:.1f} s")
print(f"Range resolution: {resR:.1f} km")
print(f"Modulation bit length: {bitLength:.1f} km")
print(f"IRI model time: {refdate_models}")
print("")
print(f"h (km)  Ne (1e11 m^-3)  dNe/Ne (SKI)  dNe/Ne (KRS)  dNe/Ne (KAI)")
for ii in range(TXaz.size*hh.size):
    print( f"{points.loc[ii:ii,'h'].values[0]:6.0f} {(plasma_params.loc[ii:ii,'ne'].values[0]/1e11):15.3f} {parerrs.loc[ii:ii,'dne1'].values[0]/plasma_params.loc[ii:ii,'ne'].values[0]:13.3f} {parerrs.loc[ii:ii,'dne2'].values[0]/plasma_params.loc[ii:ii,'ne'].values[0]:13.3f} {parerrs.loc[ii:ii,'dne3'].values[0]/plasma_params.loc[ii:ii,'ne'].values[0]:13.3f}" )

print('')
print('')
print('')


