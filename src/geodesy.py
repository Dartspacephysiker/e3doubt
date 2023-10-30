import numpy as np

d2r = np.pi/180
r2d = 180  /np.pi
WGS84_e2 = 0.00669437999014
WGS84_a  = 6378.137



def geocentric2geodeticlat(geocentriclat, degrees = True):
    """ geocentriclat is geocentric latitude (not colat). Output is geodetic latitude, WGS84 is assumed """
    if degrees:
        conv = d2r
    else:
        convv = 1.

    return -np.arctan(np.tan(geocentriclat*conv)/(WGS84_e2 - 1))/conv


def geodetic2geocentriclat(geodeticlat, degrees = True):
    """ geodeticlat is geodetic latitude (not colat). Output is geocentric latitude, WGS84 is assumed"""
    if degrees:
        conv = d2r
    else:
        conv = 1.

    return np.arctan((1 - WGS84_e2)*np.tan(geodeticlat*conv)) / conv

def geodeticheight2geocentricR(lat, height):
    """ calculate geocentric radius from geodetic latitude and height in [km] """
    N = WGS84_a/np.sqrt(1 - WGS84_e2*np.sin(lat*d2r)**2)
    
    R = np.sqrt((N + height)**2*np.cos(lat*d2r)**2 + (N*(1 - WGS84_e2) + height)**2*np.sin(lat*d2r)**2)

    return R

def geodetic2geocentricXYZ(geodeticlat, geodeticlon, height, returnR = False, degrees = True):
    """ convert geoodetic lat, lon, height to geocentric X, Y, Z. Input in degrees if degrees is True 
        height must be given in km
        The output will be given in km

        If returnR = True, the function will return ||(x, y, z)|| instead of (x, y, z) 
    """

    if degrees:
        conv = d2r
    else:
        conv = 1.

    # compute "radius of curvature in the prime vertical"
    RN = WGS84_a/np.sqrt(1 - WGS84_e2 * np.sin(geodeticlat*conv)**2)

    x = (RN + height)*np.cos(geodeticlat*conv) * np.cos(geodeticlon*conv)
    y = (RN + height)*np.cos(geodeticlat*conv) * np.sin(geodeticlon*conv)
    z = ((1 - WGS84_e2)*RN + height)*np.sin(geodeticlat*conv)

    if returnR:
        return np.linalg.norm([x, y, z])
    else:
        return x, y, z

def ECEF2geodetic(x, y, z, degrees = True):
    """ 
    convert from x, y, z ECEF to geodetic coordinates, h, lat, lon.
    returns h, lat, lon

    input should be given in km

    Using the Zhu algorithm described in:
    J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates 
    to geodetic coordinates," IEEE Transactions on Aerospace and 
    Electronic Systems, vol. 30, pp. 957-961, 1994.
    """

    conv = r2d if degrees else 1.

    # compute the lon straighforwardly
    lon = ((np.arctan2(y, x)*180/np.pi) % 360)/180*np.pi * conv

    # input params
    e = np.sqrt(WGS84_e2)
    a = WGS84_a
    b = a*np.sqrt(1 - e**2)
    r = np.sqrt(x**2 + y**2)

    # and enter the algorithm...
    l = e**2 / 2.
    m = (r / a)**2
    n = ((1 - e**2) * z / b)**2
    i = -(2 * l**2 + m + n) / 2.
    k = l**2 * (l**2 - m - n)
    q = (m + n - 4 * l**2)**3/216. + m * n * l**2
    D = np.sqrt((2 * q - m * n * l**2) * m * n * l**2 + 0j)
    beta = i/3. - (q + D)**(1/3.) - (q - D)**(1/3.) 
    t = np.sqrt(np.sqrt(beta**2 - k) - (beta + i)/2) - np.sign(m - n) * np.sqrt((beta - i) / 2)
    r0 = r / (t + l)
    z0 = (1 - e**2) * z / (t - l)
    lat = np.arctan(z0 / ((1 - e**2) * r0)) * conv
    h = np.sign(t - 1 + l) * np.sqrt((r - r0)**2 + (z - z0)**2)

    return h.real, lat.real, lon


def R_geocentricENU2geodeticENU(lat, geocentric = True):
    """ 
    returns the rotation matrix that rotates the vector[[south], [up]] from a geocentric coordinate system to a geodetic coordinate system (WGS84) 
    that is - [[south], [up]] is a vector pointing south and up in a geocentric coordinate system
    R_geoc2geod.dot([[south, [up]]]) points south and up in a geodetic coordinate system. 

    lat is latitude in geocentric coordinates if geocentric is True
    if not, it is lat geodetic coordinates

    coordinates in degrees (lat, not colat)

    # this must be checked I think....
    """

    latc = lat if geocentric else geodetic2geocentriclat(lat, degrees = True)
    latd = geocentric2geodeticlat(lat, degrees = True) if geocentric else lat

    latdiff = float(latd - latc)

    if lat > 0:
        assert latdiff >= 0 # the difference should be non-negative in the north
    else:
        assert latdiff <= 0 # ... and non-positive in the south


    return np.array([[cos(latdiff*d2r), -sin(latdiff*d2r)], [sin(latdiff*d2r), cos(latdiff*d2r)]])

def geod2geoc(gdlat, height, X, Z):
    """
    theta, r, B_th, B_r = geod2lat(gdlat, height, X, Z)

       INPUTS:    
       gdlat is geodetic latitude (not colat)
       height is geodetic height (km)
       X is northward vector component in geodetic coordinates 
       Z is downward vector component in geodetic coordinates

       OUTPUTS:
       theta is geocentric colatitude (degrees)
       r is geocentric radius (km)
       B_th is geocentric southward component (theta direction)
       B_r is geocentric radial component


    after Matlab code by Nils Olsen, DTU
    """

    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    sin_alpha_2 = np.sin(gdlat*d2r)**2
    cos_alpha_2 = np.cos(gdlat*d2r)**2

    # calculate geocentric latitude and radius
    tmp = height * np.sqrt(a**2 * cos_alpha_2 + b**2 * sin_alpha_2)
    beta = np.arctan((tmp + b**2)/(tmp + a**2) * np.tan(gdlat * d2r))
    theta = np.pi/2 - beta
    r = np.sqrt(height**2 + 2 * tmp + a**2 * (1 - (1 - (b/a)**4) * sin_alpha_2) / (1 - (1 - (b/a)**2) * sin_alpha_2))

    # calculate geocentric components
    psi  =  np.sin(gdlat*d2r) * np.sin(theta) - np.cos(gdlat*d2r) * np.cos(theta)
    
    B_r  = -np.sin(psi) * X - np.cos(psi) * Z
    B_th = -np.cos(psi) * X + np.sin(psi) * Z

    theta = theta/d2r

    return theta, r, B_th, B_r
 
def geoc2geod(theta, r, B_th, B_r):
    """
    gdlat, height, X, Z = geod2lat(theta, r, B_th, B_r)

       INPUTS:    
       theta is geocentric colatitude (degrees)
       r is geocentric radius (km)
       B_r is geocentric radial component
       B_th is geocentric southward component (theta direction)

       OUTPUTS:
       gdlat is geodetic latitude (degrees, not colat)
       height is geodetic height (km)
       X is northward vector component in geodetic coordinates 
       Z is downward vector component in geodetic coordinates


    after Matlab code by Nils Olsen, DTU
    """
    
    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    E2 = 1.-(b/a)**2
    E4 = E2*E2
    E6 = E4*E2
    E8 = E4*E4
    OME2REQ = (1.-E2)*a
    A21 =     (512.*E2 + 128.*E4 + 60.*E6 + 35.*E8)/1024.
    A22 =     (                        E6 +     E8)/  32.
    A23 = -3.*(                     4.*E6 +  3.*E8)/ 256.
    A41 =    -(           64.*E4 + 48.*E6 + 35.*E8)/1024.
    A42 =     (            4.*E4 +  2.*E6 +     E8)/  16.
    A43 =                                   15.*E8 / 256.
    A44 =                                      -E8 /  16.
    A61 =  3.*(                     4.*E6 +  5.*E8)/1024.
    A62 = -3.*(                        E6 +     E8)/  32.
    A63 = 35.*(                     4.*E6 +  3.*E8)/ 768.
    A81 =                                   -5.*E8 /2048.
    A82 =                                   64.*E8 /2048.
    A83 =                                 -252.*E8 /2048.
    A84 =                                  320.*E8 /2048.
    
    GCLAT = (90-theta)
    SCL = np.sin(GCLAT * d2r)
    
    RI = a/r
    A2 = RI*(A21 + RI * (A22 + RI* A23))
    A4 = RI*(A41 + RI * (A42 + RI*(A43+RI*A44)))
    A6 = RI*(A61 + RI * (A62 + RI* A63))
    A8 = RI*(A81 + RI * (A82 + RI*(A83+RI*A84)))
    
    CCL = np.sqrt(1-SCL**2)
    S2CL = 2.*SCL  * CCL
    C2CL = 2.*CCL  * CCL-1.
    S4CL = 2.*S2CL * C2CL
    C4CL = 2.*C2CL * C2CL-1.
    S8CL = 2.*S4CL * C4CL
    S6CL = S2CL * C4CL + C2CL * S4CL
    
    DLTCL = S2CL * A2 + S4CL * A4 + S6CL * A6 + S8CL * A8
    gdlat = DLTCL + GCLAT * d2r
    height = r * np.cos(DLTCL)- a * np.sqrt(1 -  E2 * np.sin(gdlat) ** 2)


    # magnetic components 
    psi = np.sin(gdlat) * np.sin(theta*d2r) - np.cos(gdlat) * np.cos(theta*d2r)
    X  = -np.cos(psi) * B_th - np.sin(psi) * B_r 
    Z  =  np.sin(psi) * B_th - np.cos(psi) * B_r 

    gdlat = gdlat / d2r

    return gdlat, height, X, Z
