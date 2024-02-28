#
# A simple example about how to define new radars and beam scan patterns in e3doubt.
#
#
# IV 2024
#

import e3doubt
import numpy as np
import datetime

# transmitter location in geodetic coordinates
TXloc = ('LON',78.19,15.79)

# receiver locations
RXloc = [('LON',78.19,15.79) , ('NYA',78.93,11.91), ('BAR',78.07,14.18)]

# radar carrier frequency
radarFreq  = 200e6

# transmitter duty cycle
dutyCycle = 0.25

# "Receiver duty cycle" (typically 1, but could be reduced on the core site if multipurpose codes are used)
rxduty = [1., 1., 1.]

# system noise temperature. Notice that sky-noise increases a lot when moving to lower frequencies!
Tnoise = 300.

# transmitter power
PTX = 5e6

# elevation limits for transmission and reception
mineleTrans = 30.
mineleRec = [30., 30., 30.]

# are the antennas phased-arrays? (beams widen when steered away from zenith)
phArrTrans = True
phArrRec = (True,True,True)

# transmitter beam width (full width at half maximum)
beamwidthTX = 1.

# receiver beam width, can be different for each receiver
beamwidthRX = [1., 1., 1.]

# range resolution from modulation (bit length) in km
bitLength = 1.

# transmitter beam azimuth array
TXaz = np.array([0, 90, 180, 270])

# transmitter beam elevation array
TXel = np.array([45, 45, 45, 45])

# Altitudes, where the errors are estimated (apparently, the vector should be shorter than TXaz and TXel)
hh = np.array([100., 200.,300.])

# final integrated range resolution at each alttiude
resR = np.array([1.,5.,10.])

# time resolution (the whole scan)
resT = 60.

# reference data from IRI model
refdate_models = datetime.datetime(2020,6,20,11,0,0)

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
experiment = e3doubt.Experiment(az=TXaz,el=TXel,h=hh,refdate_models=refdate_models,transmitter=TXloc,receivers=RXloc,fwhmtx=beamwidthTX,fwhmrx=beamwidthRX,radarparms=RADAR_PARMS,resR=resR)


# run IRI and MSIS
experiment.run_models()

# Calculate the parameter error estimates
parerrs = experiment.get_uncertainties(integrationsec=resT,fwhmIonSlab=NeThickness)

# coordinates of the measurement volumes
points = experiment.get_points()

# print the results
for ii in range(TXaz.size*hh.size):
    print(points.loc[ii:ii,])
    print(parerrs.loc[ii:ii,])
    print('')


