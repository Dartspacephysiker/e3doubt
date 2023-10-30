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
# from .constants import MU0,REFRE,d2r
import ppigrf
import apexpy
from datetime import datetime
from functools import reduce
from builtins import range
import warnings

from radar import *
from geodesy import geod2geoc, ECEF2geodetic, geodeticheight2geocentricR, geodetic2geocentriclat
from utils import get_supported_sites

rc('text', usetex=False)

IRIINPUTS = ['ne','Te','Ti']
DEFAULT_IONOSPHERE = {'ne':1e11,  # m^{-3}
                      'Te':1e3,   # K
                      'Ti':500}   # K

DEFAULT_RADAR_PARMS = dict(fradar=233e6,  # Radar frequency [Hz]
                           tau0=100,      # ACF time-scale [us] (IS THIS REASONABLE?)
                           dutyCycle=.25,  # Transmitter duty cycle
                           RXduty=0.8,      # Receiver duty cycle
                           Tnoise=200, # Noise temperature for receiver sites
                           Pt=5.e6,      # Devin says " The goal for the first stage implementation of the system is for 5 MW TX power 
)

# Geomagnetic reference radius:
REFRE = 6371.2 # km

class E3DUNC(object):
    """
    
    Parameters
    ---------
    v : float
        solar wind velocity in km/s
    By : float
        IMF GSM y component in nT
    Bz : float
        IMF GSM z component in nT
    tilt : float
        dipole tilt angle in degrees
    f107 : float
        F10.7 index in s.f.u.
    minlat : float, optional
        low latitude boundary of grids  (default 60)
    maxlat : float, optional
        low latitude boundary of grids  (default 89.99)


    Examples
    --------
    >>> # initialize by supplying a set of external conditions:
    >>> m = E3DUNC()
    
    >>> # make summary plot:
        
    >>> # update model vectors (tor_c, tor_s, etc.) without 
    >>> # recalculating the other matrices:
    >>> m.update_model(new_v, new_By, new_Bz, new_tilt, new_f107)

    Attributes
    ----------
    tor_c : numpy.ndarray
        vector of cos term coefficents in the toroidal field expansion
    tor_s : numpy.ndarray
        vector of sin term coefficents in the toroidal field expansion
    keys_T : list
        list of spherical harmonic wave number pairs (n,m) corresponding to elements of tor_c and tor_s 
    """

    def __init__(self, az=None, el=None, h=None,
                 refdate=datetime(2017,8,5,22,0,0),  # IRI reference date (important date)
                 fwhmres=3,
                 plasma_parms=dict(),
                 fill_IRI=True,  # If not all plasma parms provided, fill with IRI
                 dwell_times=None,
                 bit_length=None,
                 beam_width=None,
                 transmitter: str='SKI',
                 receivers: list=['SKI','KAI','KRS'],
                 fwhmtx = 2.2,
                 fwhmrx = [1.3, 1.8, 1.8],
                 radarparms = DEFAULT_RADAR_PARMS,
    ):
        """ __init__ function for E3DUNC class
        """

        print("WARNING: Haven't implemented/stored following keywords: dwell_times, bit_length, beam_width")

        ####################
        # Make sure we have the location of transmitter and receiver(s) that the user wants
        self.setup_radarconfig(transmitter,fwhmtx, receivers, fwhmrx,
                               radarparms)
    
        ####################
        # Handle input azimuths, elevations, altitudes
        self.setup_az_el_h_arrays(az, el, h)
        
        ####################
        # Handle model ionosphere
        self.refdate = refdate
        self._callable_IRI = False
        try:
            import iricore
            self._callable_IRI = True
        except:
            print("Couldn't import 'iricore' python module!")# If you don't provide 'plasma_parms' dictionary containing 'ne', 'Te', 'Te/Ti', then 'iricore' must be available to compute these!")

        self.IRI = {i:True for i in IRIINPUTS}
        self.do_call_IRI = False
        self.ionosphere = {i:DEFAULT_IONOSPHERE[i] for i in IRIINPUTS}

        # See if we need iricore
        for param in plasma_parms:
            if param in IRIINPUTS:
                self.IRI[param] = False
            else:
                raise Exception("'plasma_parms' can only contain the following parameters: {:s}".format(", ".join(IRIINPUTS)))
                
        for param in IRIINPUTS:
            self.do_call_IRI = self.do_call_IRI or self.IRI[param]


        if self.do_call_IRI:
            if not self._callable_IRI:
               for param in plasma_parms:
                   if self.IRI[param]:
                       print(f"Using default for {param}: {self.ionosphere[param]}")
   
            else:
                print("Gonna call IRI now")
                
                Te = np.zeros(self.N['pt'])*np.nan
                Ti = np.zeros(self.N['pt'])*np.nan
                Tn = np.zeros(self.N['pt'])*np.nan
                ne = np.zeros(self.N['pt'])*np.nan
                for i in range(len(self.points['az'])):
                    lat,lon,h = self.points['gdlat'][i],self.points['glon'][i],self.points['h'][i]
                    iri_out = iricore.iri(self.refdate, [h, h, 1], lat, lon, version=20)
                    
                    Te[i] = float(iri_out.etemp)
                    Ti[i] = float(iri_out.itemp)
                    Tn[i] = float(iri_out.ntemp)
                    ne[i] = float(iri_out.edens)

                    if i%10 == 0:
                        print(i)

                self.points['IRI'] = dict(ne=ne,
                                          Te=Te,
                                          Ti=Ti,
                                          Tn=Tn)
                # iricore.(self.refdate, )


    def setup_radarconfig(self, transmitter,fwhmtx, receivers, fwhmrx,
                          radarparms):
        """
        Make dictionaries containing information about transmitter, receivers, and other radar parameters
        """

        Pt, fradar, Tnoise, dutyCycle, RXduty, tau0 = (radarparms[name] for name in
                                                       ["Pt", "fradar", "Tnoise",
                                                        "dutyCycle", "RXduty", "tau0"])

        self.tx = transmitter
    
        self._sites = get_supported_sites()
    
        assert transmitter in list(self._sites.index),f"Requested transmitter site '{transmitter}' not in list of supported sites: {list(self._sites.index)}"
    
        gdlat, glon = self._sites.loc[transmitter].values
        eR, nR, uR = get_enu_vectors_cartesian(geodetic2geocentriclat(gdlat),glon,degrees=True)
        R = geodeticheight2geocentricR(gdlat, 0.) * uR
    
        txdict = {'name' :transmitter,
                  'gdlat':self._sites.loc[transmitter]['LAT(deg)'],
                  'glon' :self._sites.loc[transmitter]['LON(deg)'],
                  'ECEF' : R.ravel(),
                  'fwhm' :fwhmtx}
    
        rxdicts = []
        for rxi,rx in enumerate(receivers):
            assert rx in list(self._sites.index),f"Requested receiver site '{rx}' not in list of supported sites: {list(self._sites.index)}"
    
            gdlat, glon = self._sites.loc[rx].values
            eR, nR, uR = get_enu_vectors_cartesian(geodetic2geocentriclat(gdlat),glon,degrees=True)
            R = geodeticheight2geocentricR(gdlat, 0.) * uR
    
            rxdicts.append({'name' :rx,
                            'gdlat':self._sites.loc[rx]['LAT(deg)'],
                            'glon' :self._sites.loc[rx]['LON(deg)'],
                            'ECEF' :R.ravel(),
                            'fwhm' :fwhmrx[rxi]})
    
        self._radarconfig = {'tx'        : txdict,
                             'rx'        : rxdicts,
                             'Pt'        : Pt,
                             'fradar'    : fradar,
                             'Tnoise'    : Tnoise,
                             'RXduty'    : RXduty,
                             'dutyCycle' : dutyCycle,
                             'tau0'      : tau0,
        }


    def setup_az_el_h_arrays(self, az, el, h):

        if az is None:
            
            az = np.array([  0,  35,  69, 101, 130, 156, 180, 204, 231, 258, 288, 323,
                             0,  30,  60,  90, 120, 150, 180, 210, 240, 270, 300, 330,
                           180, 180, 180])

        if el is None:

            el = np.array([76, 73, 72, 70, 69, 67, 66, 66, 69, 71, 73, 73,
                           54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
                           76, 82.8, 90])

        if h is None:
            h = np.array([80,100,120,150,200,250,300,350,400])

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
        rECEF = get_range_line(geodetic2geocentriclat(txgdlat),
                               txglon,
                               az, el, h,
                               returnbonus=False)

        ph, pgdlat, pglon = ECEF2geodetic(*(rECEF.T))

        self.N['pt'] = len(ph)
        self.points = dict(az=az,
                           el=el,
                           h=h,
                           gdlat=pgdlat,
                           glon=pglon,
                           ph=ph,
                           xecef=rECEF[:,0],
                           yecef=rECEF[:,1],
                           zecef=rECEF[:,2])
