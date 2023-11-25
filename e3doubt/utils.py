import numpy as np
import pandas as pd
import ppigrf
from datetime import datetime
import os

basepath = os.path.dirname(__file__)

default_site_csv = os.path.abspath(os.path.join(basepath,'../','data','sites.csv'))

# read coefficient file and store in pandas DataFrame - with column names from last row of header:
colnames = ([x for x in open(default_site_csv).readlines() if x.startswith('#')][-1][1:]).strip().split(',') 

get_supported_sites = lambda fn=default_site_csv: pd.read_table(fn, skipinitialspace = True, comment = '#', sep = ',', names = colnames, index_col = [0])


def coll_freqs(nN2,nO2,nO,Te,Ti,Tn):
    """
    This returns collision frequencies from Workayehu et al (2020) Appendix A
    0. Electron-neutral coll. freq.   , Eq (A3).
    1. Ion-neutral coll. freq. for NO+, Eq (A4)
    2. Ion-neutral coll. freq. for O2+, Eq (A5)
    3. Ion-neutral coll. freq. for O+ , Eq (A6)
    
    Input
    ====
    nN2   : neutral N2 number density (m^-3)
    nO2   : neutral O2 number density (m^-3)
    nO    : neutral O number density  (m^-3)
    
    Te    : Electron temperature (K)
    Ti    : Ion temperature (K)
    Tn    : Neutral temperature (K)
    
    From Appendix A of Workayehu et al (2020):
    "The electron-neutral and ion-neutral collision frequencies are calculated using the 
     formula given by Schunk and Nagy (2009) as follows, where the three most important 
     neutral species O2, N2, and O are included"
    
    """

    #Reduced ion-neutral temperature
    Tr = (Ti+Tn)/2

    ven = 2.33e-17 * nN2 * (1 - 1.21e-4 * Te) * Te 	+ 1.82e-16 * nO2 * (1 + 3.6e-2 * np.sqrt(Te) ) * np.sqrt(Te) 	+ 8.9e-17 * nO * (1 + 5.7e-4 * Te) * np.sqrt(Te)
	
    #Ion-neutral collision frequencies
    #NO+ collision frequency
    vin1 = 4.34e-16*nN2 + 4.27e-16*nO2 + 2.44e-16*nO
	
    #O2+ collision frequency
    vin2 = 4.13e-16*nN2 + 2.31e-16*nO 	+ 2.59e-17*nO2*np.sqrt(Tr)*(1-0.073*np.log10(Tr))**2
	
    #O+ collision frequency
    vin3 = 6.82e-16*nN2 + 6.66e-16*nO2 	+ 3.67e-17*nO * np.sqrt(Tr)*(1-0.064*np.log10(Tr))**2
	
    return ven, vin1, vin2, vin3


def field_geometry_lonlat(lon,lat,date,h_km=0):
    """
    Calculate inclination, declination, and dip latitude 
    for given location(s) using IGRF.

    Example:
    --------
    inc, dec, diplat = field_geometry_lonlat()

    Parameters
    ----------
    lon : array
        longitude [deg], postiive east, of IGRF calculation
    lat : array
        geodetic latitude [deg] of IGRF calculation
    date : date(s)
        one or more dates to evaluate IGRF coefficients

    Keywords
    ----------
    h : array
        height [km] above ellipsoid for IGRF calculation

    Returns
    -------
    inc    : array
        Geomagnetic field inclination [degrees]
    dec    : array
        Geomagnetic field declination [degrees]
    diplat : array
        Geomagnetic dip latitude      [degrees]
    

    Definitions
    -----------
    Inclination : Angle made with horizontal by Earth's magnetic field lines
    Declination : Angle between geodetic north and magnetic north
    Dip latitude: Latitude of given dip angle (related to magnetic latitude?)
    S. M. Hatch
    Mar 2022
    
    """

    Be, Bn, Bu = ppigrf.igrf(lon, lat, h_km, date) # returns east, north, up

    Z = -Bu
    H = np.sqrt(Be**2+Bn**2)

    inc    = np.arctan2(Z,H)
    dec    = np.arccos(Bn/H)
    diplat = np.arctan2(Z,2*H)
    
    inc,dec,diplat = map(lambda x: np.ravel(np.rad2deg(x)), [inc,dec,diplat])

    return inc, dec, diplat


def get_perpendicular_velocity_mapping_matrix(gdlat,glon,h,
                                              mapto_h=None,
                                              apex_refh=110,
                                              refdate=datetime(2020,12,1),
                                              return_mapped_coordinates=False):
    """
    B = get_perpendicular_velocity_mapping_matrix(gdlat, glon, h, 
                                                  apex_refh=110, 
                                                  refdate=datetime(2020,12,1))

    Get matrix that maps velocity vectors (in general with field-aligned components) at a 
    given set of geographic longitudes and geodetic latitudes and altitudes to the height 
    mapto_h (default: apex_refh). 

    """

    from apexpy import Apex
    

    if mapto_h is None:
        mapto_h = apex_refh

    a = Apex(refdate,refh=apex_refh)

    # Get gdlat and glon of points at ionospheric altitude
    alat, alon = a.geo2apex(gdlat, glon, h)

    gdlatp, glonp, err = a.apex2geo(alat, alon, mapto_h)


    # get base vectors at height and at reference altitude, where 'p' mean 'prime' (i.e., at reference altitude)
    # components are geodetic east, north, and up
    f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = a.basevectors_apex(gdlat, glon, h)
    f1p, f2p, f3p, g1p, g2p, g3p, d1p, d2p, d3p, e1p, e2p, e3p = a.basevectors_apex(gdlatp, glonp, mapto_h)

    e1pe2p = np.transpose(np.stack([e1p,e2p]),axes=[1, 0, 2])  # dimension 0: ENU components; 1: e1p or e2p; 2: meas location
    d1d2 = np.stack([d1,d2]) # dimension 0: d1 or d2; 1: ENU components; 2: meas location

    # Form B matrix, which represents calculation of ve1 (=dot(v,d1)) and ve2(=dot(v,d2))
    # followed by projection to reference ionospheric height 
    # sum is over e1p/d1 and e2p/d2 vectors, so first two dimensions of B both index ENU components
    # last dimension is meas location

    # transpose to put measurement index as last index (convenient for using einsum later, I think)
    B = np.transpose(np.einsum('ij...,jk...',e1pe2p,d1d2),axes=[1,2,0])  

    if return_mapped_coordinates:
        return B, (gdlatp, glonp)

    else:
        return B
        

