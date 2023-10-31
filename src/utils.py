import numpy as np
import pandas as pd
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
