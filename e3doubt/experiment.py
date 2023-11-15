""" 
Python interface for EISCAT_3D Uncertainty Tools

This module can be used to 
1) 

MIT License

Copyright (c) 2023 Spencer M. Hatch and Ilkka Virtanen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import ppigrf
import apexpy
from datetime import datetime
from functools import reduce
from builtins import range
import warnings

from e3doubt.radar_utils import *
from e3doubt.geodesy import geoc2geod, geod2geoc, ECEF2geodetic, geodeticheight2geocentricR, geodetic2geocentriclat
from e3doubt.utils import get_supported_sites, coll_freqs

rc('text', usetex=False)

# Try to pull in importr function and robjects from rpy2
try:
    from rpy2.robjects.packages import importr
except:
    print("Couldn't import 'importr' from 'rpy2.robjects.packages'! Check rpy2 installation")
    importr = None
try:
    from rpy2 import robjects as robj
except:
    print("Couldn't import 'robjects' from rpy2! Check rpy2 installation")
    robj = None

# Pull in IRI model
try:
    import iri2016
except:
    print("Couldn't import python package 'iri2016'! Make sure it's installed")
    iri2016 = None

# Pull in MSIS model
try:
    from pymsis import msis as pmsis
except:
    print("Couldn't import python package 'iri2016'! Make sure it's installed")
    pmsis = None
    

to_floatv = lambda x: robj.vectors.FloatVector(x)
get_gd = lambda trx: (trx['gdlat'],trx['glon'])

IRIINPUTS = ['ne','Te','Ti']
DEFAULT_IONOSPHERE = {'ne':1e11,  # m^{-3}
                      'Te':1e3,   # K
                      'Ti':500}   # K

DEFAULT_RADAR_PARMS = dict(fradar=233e6,  # Radar frequency [Hz]
                           # tau0=100,      # ACF time-scale [us] (IS THIS REASONABLE?); Calculated automatically by parameterErrorEstimates.R
                           dutyCycle=.25,  # Transmitter duty cycle
                           RXduty=(0.75,1.,1.),      # Receiver duty cycle
                           Tnoise=300, # Noise temperature for receiver sites
                           Pt=3.5e6,      # Devin says " The goal for the first stage implementation of the system is for 5 MW TX power 
                           mineleTrans=30,
                           mineleRec=(30,30,30),
                           phArrTrans=True,
                           phArrRec=(True,True,True),
)

DEFAULT_AZ = np.array([  0,  35,  69, 101, 130, 156, 180, 204, 231, 258, 288, 323,
                         0,  30,  60,  90, 120, 150, 180, 210, 240, 270, 300, 330,
                       180, 180, 180])
DEFAULT_EL = np.array([76, 73, 72, 70, 69, 67, 66, 66, 69, 71, 73, 73,
                       54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
                       76, 82.8, 90])
DEFAULT_H  = np.array([80,100,120,150,200,250,300,350,400])

# Geomagnetic reference radius:
# REFRE = 6371.2 # km

class EXPERIMENT(object):
    """
    
    Parameters
    ------------------
    v : float
        solar wind velocity in km/s

    Examples
    ------------------
    >>> # initialize 
    >>> m = EXPERIMENT()
    
    >>> # make summary plot:
        

    Public attributes
    ------------------
    var : type
        description

    Private attributes
    ------------------
    
    """

    def __init__(self,
                 az=DEFAULT_AZ,
                 el=DEFAULT_EL,
                 h=DEFAULT_H,
                 refdate=datetime(2017,8,5,22,0,0),  # IRI reference date (important date)
                 fwhmres=3,
                 fill_IRI=True,  # If not all plasma parms provided, fill with IRI
                 dwell_times=None,
                 bit_length=None,
                 beam_width=None,
                 fwhmRange=3,
                 transmitter: str='SKI',
                 receivers: list=['SKI','KAI','KRS'],
                 fwhmtx = 2.1,
                 fwhmrx = [1.2, 1.7, 1.7],
                 radarparms = DEFAULT_RADAR_PARMS,
                 default_ionosphere=DEFAULT_IONOSPHERE,
    ):
        """ __init__ function for EXPERIMENT class

        fwhmRange: Range resolution in ACF decoding [km]
        
        If you wish to set values of ionospheric and atmospheric parameters manually, use the set_ionos and set_atmos methods.

        """

        print("WARNING: Haven't implemented/stored following keywords: bit_length, beam_width")

        ####################
        # Make sure we have the location of transmitter and receiver(s) that the user wants
        self._setup_radarconfig(transmitter,fwhmtx, receivers, fwhmrx,
                                radarparms)
    
        ####################
        # Handle input azimuths, elevations, altitudes
        self._setup_az_el_h_arrays(az, el, h, dwell_times)
        
        self._refdate = refdate

        ####################
        # Get ready for IRI and MSIS
        blank = np.zeros(self.N['pt'])*np.nan

        self._ionos = pd.DataFrame(
            {
                'ne':np.copy(blank),
                'Te':np.copy(blank),
                'Ti':np.copy(blank),
                'Tn':np.copy(blank),
                'O+' :np.copy(blank),
                'H+' :np.copy(blank),
                'He+':np.copy(blank),
                'O2+':np.copy(blank),
                'NO+':np.copy(blank),
                'N+' :np.copy(blank),
                'fracO+':np.copy(blank),
                'nuin':np.copy(blank)
            }
        )

        self._atmos = pd.DataFrame(
            dict(
                rhom=np.copy(blank),
                N2=np.copy(blank),
                O2=np.copy(blank),
                O=np.copy(blank),
                He=np.copy(blank),
                H=np.copy(blank),
                Ar=np.copy(blank),
                N=np.copy(blank),
                AnomO=np.copy(blank),
                NO=np.copy(blank),
                Tn=np.copy(blank)
            )
        )

        self._isCallable_IRI = iri2016 is not None
        self._called_IRI = False

        self._IRI = {i:True for i in IRIINPUTS}
        self._do_call_IRI = False
        self._default_ionosphere = {p:default_ionosphere[p] for p in IRIINPUTS}

        self._isCallable_MSIS = pmsis is not None
        self._called_MSIS = False

        self._isCallable_R = (importr is not None) and (robj is not None)
        assert self._isCallable_R,"No point in going further, R can't be called from python"

        # These members are switched to "True" when
        # user changes 
        self._isChanged_atmos = False
        self._isChanged_ionos = False

        ####################
        # Handle R via rpy2
        self._ISgeometry = None
        self._rparms = None
        self.init_R()
        self._init_unc_parms()

        self._dfunc = None
        self._dfunc_kw = None

    def run_models(self):
        ####################
        # Handle model ionosphere

        self._run_IRI()

        ####################
        # Handle model atmosphere

        self._run_MSIS()

        ####################
        # Calculate collision frequency
        self._calc_collfreq()

        print("IRI output, MSIS output, and collision frequencies stored in EXPERIMENT._ionos, EXPERIMENT._atmos, and EXPERIMENT._ionos['nuin']")
        print("Next, you can perform uncertainty calculations with EXPERIMENT.calc_uncertainties()")


    def _setup_radarconfig(self, transmitter,fwhmtx, receivers, fwhmrx,
                           radarparms):
        """
        Make dictionaries containing information about transmitter, receivers, and other radar parameters
        """

        Pt, fradar, Tnoise, dutyCycle, RXduty = (radarparms[name] for name in
                                                 ["Pt", "fradar", "Tnoise",
                                                  "dutyCycle", "RXduty"])
        mineleTrans = radarparms['mineleTrans']
        mineleRec = radarparms['mineleRec']
        phArrTrans = radarparms['phArrTrans']
        phArrRec = radarparms['phArrRec']

        self._sites = get_supported_sites()
    
        assert transmitter in list(self._sites.index),f"Requested transmitter site '{transmitter}' not in list of supported sites: {list(self._sites.index)}"
    
        gdlat, glon = self._sites.loc[transmitter].values
        gclat = geodetic2geocentriclat(gdlat)
        eR, nR, uR = get_enu_vectors_cartesian(gclat,glon,degrees=True)
        R = geodeticheight2geocentricR(gdlat, 0.) * uR
    
        txdict = {'name'  :transmitter,
                  'gdlat' :gdlat,
                  'gclat' :gclat,
                  'glon'  :glon,
                  'ECEF'  : R.ravel(),
                  'fwhm'  :fwhmtx,
                  'minele':mineleTrans,
                  'phArr':phArrTrans,
        }
    
        rxdicts = []
        for rxi,rx in enumerate(receivers):
            assert rx in list(self._sites.index),f"Requested receiver site '{rx}' not in list of supported sites: {list(self._sites.index)}"
    
            gdlat, glon = self._sites.loc[rx].values
            gclat = geodetic2geocentriclat(gdlat)
            eR, nR, uR = get_enu_vectors_cartesian(geodetic2geocentriclat(gdlat),glon,degrees=True)
            R = geodeticheight2geocentricR(gdlat, 0.) * uR
    
            rxdicts.append({'name' :rx,
                            'gdlat':gdlat,
                            'gclat':gclat,
                            'glon' :glon,
                            'ECEF' :R.ravel(),
                            'fwhm' :fwhmrx[rxi],
                            'minele':mineleRec[rxi],
                            'phArr':phArrRec[rxi]})
    
        self._radarconfig = {'tx'        : txdict,
                             'rx'        : rxdicts,
                             'Pt'        : Pt,
                             'fradar'    : fradar,
                             'Tnoise'    : Tnoise,
                             'RXduty'    : RXduty,
                             'dutyCycle' : dutyCycle,
                             # 'tau0'      : tau0,
        }


    def _setup_az_el_h_arrays(self, az, el, h, dwell_times):

        self.N = dict(az = az.size,
                      el = el.size,
                      h = h.size,
                      tx = 1,
                      rx = len(self._radarconfig['rx']))
            
        if az.size != el.size:

            if az.size == 1 and el.size > 1:
                az = np.broadcast_to(az, el.shape)
            elif az.size > 1 and el.size == 1:
                el = np.broadcast_to(el, az.shape)

            else:

                shape = (az.size, el.size)

                az,el = map(lambda x: np.broadcast_to(x, shape).ravel(), [az[:,np.newaxis], el[np.newaxis,:]])


        if az.size != h.size:

            if az.size == 1 and h.size > 1:
                az = np.broadcast_to(az, h.shape)
                el = np.broadcast_to(el, h.shape)
            elif az.size > 1 and h.size == 1:
                h = np.broadcast_to(h, az.shape)

            else:

                shape = (az.size, h.size)

                az,el,h = map(lambda x: np.broadcast_to(x, shape).ravel(), [az[:,np.newaxis],
                                                                            el[:,np.newaxis],
                                                                            h[np.newaxis,:]])
            
        ####################
        # Get these points in ECEF coordinates
        txgdlat, txglon = self._sites.loc[self._radarconfig['tx']['name']]
        rECEF = get_range_line(txgdlat,
                               txglon,
                               az, el, h,
                               returnbonus=False)

        ph, pgdlat, pglon = ECEF2geodetic(*(rECEF.T))
        pgclat = geodetic2geocentriclat(pgdlat)

        self.N['pt'] = len(ph)

        if dwell_times is None:
            dwell_times = np.ones(self.N['pt'])

        # Normalize dwell_times so that sum is one.
        # Idea is that user provides a single total integration time, and that time is
        # divided among the beams according to the fractions implied by dwell_times.
        dwell_times = dwell_times/np.sum(dwell_times)  

        self._points = pd.DataFrame(
            dict(
                az=az,
                el=el,
                h=h,
                gdlat=pgdlat,
                gclat=pgclat,
                glon=pglon,
                # ph=ph,
                xecef=rECEF[:,0],
                yecef=rECEF[:,1],
                zecef=rECEF[:,2],
                dwell_time=dwell_times,
            )
        )


    def _run_IRI(self):

        self._do_call_IRI = True

        if self._do_call_IRI:
            if not self._isCallable_IRI:
               # for parm in plasma_parms:
               #     if self._IRI[parm]:
               #         print(f"Using default for {parm}: {self._default_ionosphere[parm]}")
                assert 2<0,"Can't call IRI!"
            else:
                print("Gonna call IRI now")
                
                for i in range(len(self._points['az'])):
                    lat,lon,h = self._points['gdlat'][i],self._points['glon'][i],self._points['h'][i]

                    iri_out = iri2016.IRI(self._refdate, [h, h, 1], lat, lon)
                    
                    self._ionos.loc[i,'ne'] = iri_out.ne.values[0]
                    self._ionos.loc[i,'Te'] = iri_out.Te.values[0]
                    self._ionos.loc[i,'Ti'] = iri_out.Ti.values[0]
                    self._ionos.loc[i,'Tn'] = iri_out.Tn.values[0]
                    self._ionos.loc[i,'fracO+'] = (iri_out['nO+']/iri_out['ne']).values[0]
                    self._ionos.loc[i,'O+' ] = iri_out['nO+' ].values[0]
                    self._ionos.loc[i,'H+' ] = iri_out['nH+' ].values[0]
                    self._ionos.loc[i,'He+'] = iri_out['nHe+'].values[0]
                    self._ionos.loc[i,'O2+'] = iri_out['nO2+'].values[0]
                    self._ionos.loc[i,'NO+'] = iri_out['nNO+'].values[0]
                    self._ionos.loc[i,'N+' ] = iri_out['nN+' ].values[0]

                    if i%10 == 0:
                        print(i)

                self._called_IRI = True


    def _run_MSIS(self):

        if self._isCallable_MSIS:
            print("Gonna call MSIS now")
                
            dates = [self._refdate]*self.N['pt']
            lons,lats,alts = self._points[['glon','gdlat','h']].values.T
            out = pmsis.run(dates, lons, lats, alts) #f107s, f107as, aps)

            # From pymsis documentation
            # ndarray (ndates, nlons, nlats, nalts, 11) or (ndates, 11)
            #     | The data calculated at each grid point:
            #     | [Total mass density (kg/m3)
            #     | N2 # density (m-3),
            #     | O2 # density (m-3),
            #     | O # density (m-3),
            #     | He # density (m-3),
            #     | H # density (m-3),
            #     | Ar # density (m-3),
            #     | N # density (m-3),
            #     | Anomalous oxygen # density (m-3),
            #     | NO # density (m-3),
            #     | Temperature (K)]
            
            # rhom, N2, O2, O, He, H, Ar, N, AnomO, NO, Tn = out.T

            for ic,col in enumerate(self._atmos.columns):
                self._atmos.loc[self._atmos.index,col] = out[:,ic]

            self._called_MSIS = True

        else:
            print("Sorry, can't get MSIS!")

            
    def init_R(self):
        
        while True:

            self._ISgeometry = importr("ISgeometry")

            self._isCallable_R = True
            # Try to pull in ISgeometry package
            # try:
            #     ISgeometry = importr("ISgeometry")
                
            # except:
            #     print("Couldn't import 'ISgeometry' R package! Will atte")
            #     break
    
            break


    def _calc_collfreq(self):
        """
        Assume that the right thing to do is to calculate an weighted average collision frequency, where the weights are the abundances of NO+, O2+, and O+
        """
        
        ven, vinNOp, vinO2p, vinOp = coll_freqs(self._atmos['N2'],self._atmos['O2'],self._atmos['O'],
                                                self._ionos['Te'],self._ionos['Ti'],self._atmos['Tn'])
        
        # nuin = vinNOp*self._ionos['NO+']+vinO2p*self._ionos['O2+']+vinOp*self._ionos['O+']
        
        # iri_ionmass = {'O+': 15.999,  # in AMU
        #                'H+': 1.00784,
        #                'He+': 4.002602,
        #                'O2+': 15.999*2,
        #                'NO+': 14.0067+15.999,
        #                'N+': 14.0067}
        # rho_mass = (self._ionos['NO+']*iri_ionmass['NO+']+self._ionos['O2+']*iri_ionmass['O2+']+self._ionos['O+']*iri_ionmass['O+'])/(self._ionos['NO+']+self._ionos['O2+']+self._ionos['O+'])
        self._ionos['nuin'] = (vinNOp*self._ionos['NO+']+\
                               vinO2p*self._ionos['O2+']+\
                               vinOp *self._ionos['O+']   )/self._ionos['ne']


    def _init_unc_parms(self,
                       # Tnoise: int=300,
                       # Pt: float=3.5e6,
                       # locTrans: tuple=(69.45719, 20.46667),
                       # locRec: tuple=((69.45719, 20.46667),
                       #                (68.157,19.45),
                       #                (68.3175,22.48325)),
                       # fwhmTrans: float=2.1,
                       # fwhmRec: tuple=(1.2, 1.7, 1.7),
                       # RXduty: tuple=(.75,1,1),
                       # mineleTrans: int=30,
                       # mineleRec: tuple=(30,30,30),
                       # fradar: float=233e6,
                       # phArrTrans: bool=True,
                       # phArrRec: tuple=(True,True,True),
                       # dutyCycle: float=0.25,
    ):
        """
        From I. Virtanen's r documentation
    
        # Calculate plasma parameter error estimates in the given point with the given radar system
        # and plasma parameters
        #
        # INPUTS:
        #   lat          Geocentric latitude of the measurement volume [deg north]
        #   lon          Geocentric longitude of the measurement volume [deg east]
        #   alt          Altitude of the measurement volume [km]
        #   Ne           Electron density [m^-3]
        #   Ti           Ion temperature [K]
        #   Te           Electron temperature [K]
        #   Coll         Ion-neutral collision frequency [Hz]
        #   Comp         Ion composition (fraction of ion with mass pm0[2] out of the total ion number density)
        #   fwhmRange    Range resolution in ACF decoding [km]
        #   resR         Range resolution in the plasma parameter fit [km]
        #   intTime      Integration time [s], default 10
        #   pm0          Ion masses [amu], default c(30.5,16)
        #   hTeTi        Altitude, below which Te=Ti is assumed, [km], default 110
        #   Tnoise       Receiver noise temperature [K], default 300
        #   Pt           Transmitter power(s) [W], default 3.5e6
        #   locTrans     Transmitter location(s), list of lat, lon, [height] in degress [km], default list(c(69.34,20.21))
        #   locRec       Receiver locations, list of lat, lon, [height] in degrees [km], default list(c(69.34,20.21),c(68.48,22.52),c(68.27,19.45))
        #   fwhmTrans    Transmitter beam width(s) (at zenithg for phased-arrasy), [full width at half maximu, degrees], default c(2.1)
        #   fwhmRec      Receiver beam widhts (at zenith for phased-arrays) [full width at half maximum, degrees], default c(1.2,1.7,1.7)
        #   RXduty       "Receiver duty cycle" for each receiver, default c(.75,1,1)
        #   mineleTrans  Elevation limit of the transmitter [deg], default c(30)
        #   mineleRec    Elevation limit of the receivers [deg], default c(30,30,30)
        #   fradar       Radar system carrier frequency [Hz], default 233e6
        #   phArrTrans   Logical is (are) the transmitter(s) phased array(s)? Default c(T)
        #   phArrRec     Logical, are the receivers phased-arrays? A vector with a value for each receiver. Default c(T,T,T)
        #   fwhmIonSlab  Thickness of the ion slab that causes self-noise [km], default 100
        #   dutyCycle    Transmitter duty cycle, default 0.25
        
        """
    
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("Warning! EXPERIMENT.init_unc_parms doesn't ... Wait, never mind. This does it _all_.")
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("")

        tx = self._radarconfig['tx']
        rxs = self._radarconfig['rx']
            
        Tnoise = self._radarconfig['Tnoise']
        Pt = self._radarconfig['Pt']

        locTrans = get_gd(tx)
        locRec = tuple([get_gd(rx) for rx in rxs])

        fwhmTrans = tx['fwhm']
        fwhmRec = tuple([rx['fwhm'] for rx in rxs])
        RXduty = self._radarconfig['RXduty']

        mineleTrans = self._radarconfig['tx']['minele']
        mineleRec = tuple([rx['minele'] for rx in rxs])

        fradar = self._radarconfig['fradar']

        phArrTrans = tx['phArr']
        phArrRec = tuple([rx['phArr'] for rx in rxs])

        dutyCycle = self._radarconfig['dutyCycle']

        # store a copy of dat
        uncparms = dict(
            # pm0=pm0,
            # hTeTi=hTeTi,
                        Tnoise=Tnoise,
                        Pt=Pt,
                        locTrans=locTrans,
                        locRec=locRec,
                        fwhmTrans=fwhmTrans,
                        fwhmRec=fwhmRec,
                        RXduty=RXduty,
                        mineleTrans=mineleTrans,
                        mineleRec=mineleRec,
                        fradar=fradar,
                        phArrTrans=phArrTrans,
                        phArrRec=phArrRec,
                        # fwhmIonSlab=fwhmIonSlab,
                        dutyCycle=dutyCycle)

        rparms = dict()
    
        # if Tnoise is not None:
        rparms['Tnoise'] = Tnoise
        
        # if Pt is not None:
        rparms['Pt'] = Pt
        
        # if locTrans is not None:
        rparms['locTrans'] = to_floatv(locTrans)
        
        # if locRec is not None:
        rparms['locRec'] = robj.vectors.ListVector({f"rec{i+1}":to_floatv(rec) for i,rec in enumerate(locRec)})
        
        # if fwhmTrans is not None:
        rparms['fwhmTrans'] = to_floatv((fwhmTrans,))
        
        # if RXduty is not None:
        rparms['RXduty'] = to_floatv(RXduty)    
        
        rparms['mineleTrans'] = to_floatv((mineleTrans,))
        rparms['mineleRec'] = to_floatv(mineleRec)
        
        rparms['fradar'] = fradar
        
        rparms['phArrTrans'] = robj.vectors.BoolVector((phArrTrans,))
        rparms['phArrRec'] = robj.vectors.BoolVector(phArrRec)
        rparms['dutyCycle'] = dutyCycle

        self._rparms = rparms
        self._uncparms = uncparms
        

    def calc_uncertainties(self,fwhmRange=2,resR=None,integrationsec=10,
                           force_recalc=False,
                           pm0: tuple = (30.5,16),
                           hTeTi: int=110,
                           fwhmIonSlab: int=100,
):
        """
        resR should be at least as large as fwhmRange

        fwhmRange ~ bit length of code
        resR ~ after integration, should be integer multiple of bit length

        Typical EISCAT analysis uses a more-or-less fixed set of range resolutions

        have code take stock of which inputs the user has changed. For example, if only integ time or dutyCycle changes, no need to re-run R code, just scale outputs. 

        integrationsec: Total integration time divided among all EXPERIMENT.N['pt'] beams, with time apportioned according to dwell_times

        """

        if resR is None:
            resR = fwhmRange

        # Check if we need to run a full calculation of uncertainties
        
        # Full calculation not needed if previous run exists and integrationsec is the only thing that has changed
        isExisting_run = self._dfunc is not None
        isChanged_parms = self._isChanged_ionos or self._isChanged_atmos
        if force_recalc or (not isExisting_run):
            do_run = True
        else:

            do_run = False
            # If any of these are different between this call and the previous, do a full recalculation
            RERUN_KEYS = ['fwhmRange','resR','pm0','hTeTi','fwhmIonSlab']

            # What has changed?

            dfunc_kw = dict(fwhmRange=fwhmRange,
                            resR=resR,
                            integrationsec=integrationsec,
                            pm0=pm0,
                            hTeTi=hTeTi,
                            fwhmIonSlab=fwhmIonSlab,
            )
            
            different_keys = []
            
            for k in dfunc_kw:
                
                if dfunc_kw[k] != self._dfunc_kw[k]:
                    different_keys.append(k)

            if len(different_keys) > 0:
                print("differences between this run and last run: ", different_keys)
                print(f"{'Key':20s} : {'Existing':20s} {'Requested':20s}")
                for k in different_keys:
                    print(f"{k:20s} : {str(self._dfunc_kw[k]):20s} {str(dfunc_kw[k]):20s}")

                if any([k in different_keys for k in RERUN_KEYS]):
                    do_run = True
                    print("Re-running uncertainty calculation with new parameters!")

                elif 'integrationsec' in different_keys:
                    do_run = False
                   
                    dtau = fwhmRange*1000/3e8*2
                    dutyCycle = self._radarconfig['dutyCycle']

                    integfac = np.sqrt(dtau/dutyCycle/integrationsec)

                    dtau_previous = self._dfunc_kw['fwhmRange']*1000/3e8*2
                    integfac_previous = np.sqrt(dtau_previous/dutyCycle/self._dfunc_kw['integrationsec'])

                    scale = integfac/integfac_previous
                    
                    print(f"Returning results from before, but with uncertainties scaled by a factor of {scale:.5f}")
                    dfunc = self._dfunc.copy()
                    for c in dfunc:
                        dfunc.loc[:,c] = dfunc[c].values*scale

                    return dfunc

            else:
                    
                return self._dfunc

        if do_run:

            builderarr = np.zeros(self.N['pt'])*np.nan
            
            Ndict = self.N['rx']+1  # Number of uncertainty estimates is "N receivers" plus one for multistatic estimate
            dfunc = dict() #data frame of uncertainties
            
            labels = [str(i+1) if i < self.N['rx'] else 'multi' for i in range(Ndict)]
            
            # hTeTi = self._rparms['hTeTi']
            
            runparms = dict(hTeTi=hTeTi,
                                fwhmIonSlab=fwhmIonSlab,
                                pm0=to_floatv(pm0))
            for key,var in self._rparms.items():
                runparms[key] = var
            
            parmnames = ['ne','Te','Ti','Vi']
            nelabels = ['dne'+lab for lab in labels]
            Telabels = ['dTe'+lab for lab in labels]
            Tilabels = ['dTi'+lab for lab in labels]
            Vilabels = ['dVi'+lab for lab in labels]
            for parm in parmnames:
                
                for i in range(Ndict):
                    dfunc['d'+parm+labels[i]] = np.copy(builderarr)
            
            dfunc = pd.DataFrame(dfunc)
            
            getcols = ['gclat','glon','h','dwell_time']
            ionocols = ['ne','Ti','Te','fracO+','nuin']
            ##############################
            # Now get uncertainties for everything
            for i,point in self._points.iterrows():
            
                if i%10 == 0:
                    print(i)
            
                gclat, glon, h, dwellT = point[getcols]
            
                Ne, Ti, Te, fracOp, nuin = self._ionos.iloc[i][ionocols]
            
                intT = dwellT * integrationsec  # integration time for this beam
                out = self._ISgeometry.parameterErrorEstimates(
                    gclat, glon, h, Ne, Ti, Te, nuin, fracOp, fwhmRange,
                    resR,
                    intT,
                    **runparms)
            
                los = out[0]
                dNes = [list(los[i])[0] for i in range(len(los))]
                dTis = [list(los[i])[1] for i in range(len(los))]
            
                dNes.append(list(out[1])[0])
                dTis.append(list(out[1])[1])
            
                if h > hTeTi:
                    # got dNe, dTi, dTe, dVi
                    dTes = [list(los[i])[2] for i in range(len(los))] 
                    dVis = [list(los[i])[3] for i in range(len(los))]
            
                    dTes.append(list(out[1])[2])
                    dVis.append(list(out[1])[3])
            
                else:
                    # got dNe, dTi, dVi
                    dTes = dTis
                    dVis = [list(los[i])[2] for i in range(len(los))]
                        
                    dVis.append(list(out[1])[2])
            
                dfunc.loc[i,nelabels] = dNes
                dfunc.loc[i,Telabels] = dTes
                dfunc.loc[i,Tilabels] = dTis
                dfunc.loc[i,Vilabels] = dVis
                
            # store dfunc and last run options
            self._dfunc = dfunc
            self._dfunc_kw = dict(fwhmRange=fwhmRange,
                                  resR=resR,
                                  integrationsec=integrationsec,
                                  pm0=pm0,
                                  hTeTi=hTeTi,
                                  fwhmIonSlab=fwhmIonSlab,
                                  )
            
            return dfunc


    def get_los_vectors(self,
                        coordsys='enu',
                        SANITYCHECKENU=True):
        """
        Get "line-of-sight" vectors for each point and transmitter/receiver pair.

        coordsys: must be one of 'enu' or 'ecef'
        """
        
        assert coordsys in ['enu','ecef']

        loslist = list()
        txrxlist = list()       # keep track of which pair
        for i,rx in enumerate(self._radarconfig['rx']):

            if coordsys == 'ecef':
                los = get_los_vector_ecef(self._points[['xecef','yecef','zecef']].values.T,
                                          self._radarconfig['tx']['ECEF'],
                                          rx['ECEF'])

            elif coordsys == 'enu':
                out = get_los_vector_enu(self._points[['xecef','yecef','zecef']].values.T,
                                         self._radarconfig['tx']['ECEF'],
                                         rx['ECEF'],
                                         return_coords=SANITYCHECKENU)
    
                if SANITYCHECKENU:
                    los, gdlat, h = out
    
                    # make sure calculations of geodetic altitude differ by less than 1 km
                    assert np.all(np.abs(self._points['h']-h) < 1)
    
                    # make sure calculations of geodetic latitude differ by less than .02
                    assert np.all(np.abs(gdlat-self._points['gdlat']) < 0.02)
                    
                else:
                    los = out

            txrxlist.append(self._radarconfig['tx']['name']+rx['name'])
            loslist.append(los)

        return loslist, txrxlist
        

    def velocity_cov_matrix(self,
                            los=None,
                            dv=None,
                            Cminv=0.,
                            coordsys='enu'):
        """
        los     : np.ndarray, shape (P, 3, N) with P number of transmitter-receiver pairs and N number of points
                matrix of line-of-sight vectors
    
        dv      : np.ndarray, shape (P, N)
                uncertainty of each los velocity estimate (assumed to be independent of the other estimates)
    
        Cminv   : UNUSED
                could be used to include effects of regularization of some sort
    
        Returns
        ==========
        covmats : np.ndarray, shape (3, 3, N)
                covariance matrices of each velocity estimate
        """
    
        if los is None:
            los, txrxpairs = self.get_los_vectors(coordsys=coordsys)
            los = np.stack(los)

        if dv is None:
            dv = self._dfunc[['dVi1','dVi2','dVi3']].values.T

        assert los.ndim == 3 and dv.ndim == 2
        assert (los.shape[0] == dv.shape[0]) & (los.shape[2] == dv.shape[1]) & (los.shape[1] == 3)
    
        P,_,N = los.shape
    
        Cminv = 0.
        covmat = []
        for n in range(N):
            covmat.append( np.linalg.inv(los[:,:,n].T@np.diag(1/dv[:,n])@los[:,:,n] + Cminv) )
    
    
        return np.transpose(np.stack(covmat),axes=[1,2,0])
    
    
    def get_point_df(self):

        return self._points

    def get_radarconfig(self):

        return self._radarconfig


    def get_ionos(self, name=None):

        if name is None:
            return self._ionos
        else:
            # Find out if user has made invalid request
            return _get_valids(name,self._ionos)


    def get_atmos(self, name=None):

        if name is None:
            return self._ionos
        else:
            # Find out if user has made invalid request
            return _get_valids(name,self._atmos)


    def set_ionos(self, name, values):

        self._ionos.loc[:,name] = values

        self._isChanged_ionos = True


    def set_atmos(self, parm=None):

        self._atmos.loc[:,name] = values

        # Record that we've changed atmosphere
        self._isChanged_atmos = True


def cartesian_to_spherical_with_position(x, y, z, vx, vy, vz,
                                         return_coords=True):
    """
    returns array of spherical components of vector given by vx, vy, and vz with shape (3,N)

    Written mostly by ChatGPT3.5 (for fun)!
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    vr = (vx * x + vy * y + vz * z) / r
    vtheta = (vx * np.cos(phi) * np.cos(theta) + vy * np.sin(phi) * np.cos(theta) - vz * np.sin(theta))
    vphi = (-vx * np.sin(phi) + vy * np.cos(phi))

    if return_coords:
        return np.vstack([vr, vtheta, vphi]), np.vstack([r, theta, phi])
    else:
        return np.vstack([vr, vtheta, vphi])


# def spherical_to_cartesian_with_position(r, th, phi, vr, vth, vphi,
#                                          return_coords=True):
#     """
#     returns array of ECEF components of vector given by vr, vtheta, and vphi with shape (3,N)
#     """

#     thr = np.deg2rad(th)
#     phir = np.deg2rad(phi)

#     x = r * np.sin(thr) * np.cos(phir)
#     y = r * np.sin(thr) * np.sin(phir)
#     z = r * np.cos(thr)

#     vx = 

#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arccos(z / r)
#     phi = np.arctan2(y, x)

#     vr = (vx * x + vy * y + vz * z) / r
#     vtheta = (vx * np.cos(phi) * np.cos(theta) + vy * np.sin(phi) * np.cos(theta) - vz * np.sin(theta))
#     vphi = (-vx * np.sin(phi) + vy * np.cos(phi))

#     if return_coords:
#         return np.vstack([vr, vtheta, vphi]), np.vstack([r, theta, phi])
#     else:
#         return np.vstack([vr, vtheta, vphi])


def get_los_vector_ecef(p, t, r,
                   normalize=True):
    """
    x,y,z    : ECEF coordinates of point(s)                (array, shape: 3, or (3,N))
    t        : x, y, and z ECEF coordinates of transmitter (array, shape: 3,)
    r        : x, y, and z ECEF coordinates of receiver    (array, shape: 3,)

    returns array of  "line-of-sight" vector(s) with shape (3,N)

    Equation 1 in Virtanen et al (2014, doi 10.1002/2014JA020540)
    """
    
    x, y, z = p
    kt = np.vstack([x-t[0], y-t[1], z-t[2]]) #Vector pointing from transmitter to point
    kr = np.vstack([r[0]-x, r[1]-y, r[2]-z]) #Vector pointing from point to receiver

    ks = kr-kt

    if normalize:
        ks = ks/np.linalg.norm(ks,axis=0)

    return ks


def get_los_vector_enu(p, t, r,
                       normalize=True,
                       return_coords=False):
    
    los = get_los_vector_ecef(p, t, r,
                              normalize=normalize)

    los_rtp, rtp = cartesian_to_spherical_with_position(*p, *los)

    gdlat, h, losn, losd = geoc2geod(np.rad2deg(rtp[1]), rtp[0], los_rtp[1], los_rtp[0])

    los_enu = np.vstack([los_rtp[2],losn, -losd])

    if return_coords:
        return los_enu, gdlat, h
    else:
        return los_enu


# los = get_los_vector_ecef(e3du.points[['xecef','yecef','zecef']].values.T,
#                           e3du._radarconfig['tx']['ECEF'],
#                           e3du._radarconfig['rx'][0]['ECEF'])

# los_rtp, rtp = cartesian_to_spherical_with_position(*e3du.points[['xecef','yecef','zecef']].values.T,
#                                                     *los)


def _get_valids(x,df):
    """
    valid: list of valid strings
    x    : list of parameters requested by user
    """
    invalids = []
    valids = df.columns
    if isinstance(x,str):
        x = [x]

    for y in x:
        if y not in valids:
            invalids.append(y)

    if len(invalids) > 0:
        print("Requested invalid parameter(s): ["+", ".join(inv)+"]")
        print("Valid parameters are ["+", ".join(valids)+"]")

        return None
    else:
        return df[x]



