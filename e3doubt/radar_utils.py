import numpy as np
from e3doubt.vectors import dotprod,crossprod,vecmag
from e3doubt.geodesy import geodeticheight2geocentricR,geocentric2geodeticlat,geod2geoc
import warnings

# Geomagnetic reference radius:
RE = 6371.2 # km

def point_vector_cartesian(gclat,gclon,az,el,degrees=True):
    """
    gclat and gclon are geocentric latitude and longitude.
    az and el are azimuth and elevation.

    Output is a vector pointing from the specified (lat,lon) point 
    in the direction of the given azimuth and elevation, in Cartesian 
    (Earth-centered, Earth-fixed, or ECEF) coordinates.

    NOTE! This assumes that the earth is a perfect sphere, and that your 
    (lat,lon) point is at a height of 0 km.

    # valme = "np.random.uniform(size=10)";
    # gclat,gclon,az,el = eval(valme),eval(valme),eval(valme),eval(valme)


    Example
    =======
    With gclat = 45°, az = 180°, el = 45°, our pointing vector should be
    phat = cos(phi)*xhat + sin(phi)*yhat — right? Think about it!

    If gclon (= phi) = 0°, phat = xhat.
    print(point_vector_cartesian(45,0,180,45))

    Spencer Mark Hatch
    2022/03/22
    """

    d = dict(gclat=gclat,gclon=gclon,az=az,el=el)
    d = _make_equal_length_dict(d,DEBUG=False)
    gclat,gclon,az,el = d['gclat'],d['gclon'],d['az'],d['el']

    if not degrees:
        latr,lonr,azr,elr = gclat,gclon,az,el
    else:
        latr,lonr,azr,elr = np.deg2rad([gclat,gclon,az,el])


    colatr = np.pi/2-latr

    sa = np.sin(azr)    # sa = "sine alpha"
    sb = np.sin(elr)    # sb = "sine beta"  (beta  = elevation )
    st = np.sin(colatr) # st = "sine theta" (theta = colatitude)
    sp = np.sin(lonr)   # sp = "sine phi"   (phi   = longitude )
        
    ca = np.cos(azr) 
    cb = np.cos(elr) 
    ct = np.cos(colatr) 
    cp = np.cos(lonr)

    px = sb*st*cp-cb*sa*sp-cb*ca*ct*cp
    py = sb*st*sp+cb*sa*cp-cb*ca*ct*sp
    pz = sb*ct+cb*ca*st

    return np.vstack([px,py,pz]).T


def height_to_range(h_km,el,degrees=True):
    """
    Get the distance from a point on the earth's (assumed perfectly 
    spherical) surface to a particular altitude given the elevation.

    Spencer Mark Hatch
    2022/03/22
    """

    if degrees:
        elr = np.deg2rad(el)
    else:
        elr = el

    return np.sqrt(RE**2 * np.sin(elr)**2 + 2*RE*h_km + h_km**2) - RE*np.sin(elr)


def _make_equal_length_dict(d,DEBUG=False):
    for key1,item1 in d.items():
        for key2,item2 in d.items():
            if DEBUG:
                print(f"{key1}, {key2}")
            item1, item2 = _make_equal_length(item1,item2,DEBUG=DEBUG)
            d[key1] = item1
            d[key2] = item2
            
    return d

def _make_equal_length(a,b,DEBUG=False):
    ahas = hasattr(a,"__len__")
    bhas = hasattr(b,"__len__")
    alen = len(a) if ahas else 1
    blen = len(b) if bhas else 1
    agtr = alen > blen
    bgtr = alen < blen

    if DEBUG:
        print(f"len(a), len(b) = {alen}, {blen}")
    if agtr:
        assert blen == 1
        b = np.array([b]*alen).reshape(a.shape)
    elif bgtr:
        assert alen == 1
        a = np.array([a]*blen).reshape(b.shape)

    return a,b


def get_geodetic_enu_vectors_in_ECEF(gdlatRec, glonRec, hRec=0.):

    # local enu vectors at transceiver
    e_enu = np.array([1,0,0])
    n_enu = np.array([0,1,0])
    u_enu = np.array([0,0,1])

    thetaRec, rRec, n_theta, n_r = geod2geoc(gdlatRec, hRec, n_enu[1], -n_enu[2])
    thetaRec, rRec, u_theta, u_r = geod2geoc(gdlatRec, hRec, u_enu[1], -u_enu[2])
    
    gclatRec = 90.-thetaRec

    # geodetic e,n,u vectors with spherical enu (phi,-theta,r) components
    e_sphenu = np.array([1,0,0])
    n_sphenu = np.array([0,-n_theta,n_r])
    u_sphenu = np.array([0,-u_theta,u_r])

    Rvec_SPHENU_TO_ECEF = ECEFtosphericalENUMatrix(glonRec, gclatRec).squeeze().T

    # geodetic e,n,u vectors with Cartesian ECEF components
    eR = (Rvec_SPHENU_TO_ECEF@e_sphenu)[:,np.newaxis].T
    nR = (Rvec_SPHENU_TO_ECEF@n_sphenu)[:,np.newaxis].T
    uR = (Rvec_SPHENU_TO_ECEF@u_sphenu)[:,np.newaxis].T
    
    return eR, nR, uR


def get_enu_vectors_cartesian(gclat,gclon,degrees=True):
    """
    Get east, north, up vectors for a given (lat,lon) pair or a list 
    of (lat,lon) pairs in Cartesian (Earth-centered, Earth-fixed, or 
    ECEF) coordinates.

    Spencer Mark Hatch
    2022/03/22
    """
    
    warnings.warn("This thing assumes geocentric coordinates are right! In particular, it assume that the geocentric radial unit vector is the same as the geodetic 'up' vector, which is of course not totally true!")

    gclat,gclon = _make_equal_length(gclat,gclon)

    if degrees:
        latr, lonr = np.deg2rad([gclat,gclon])
    else:
        latr, lonr = gclat,gclon

    colatr = np.pi/2-latr

    st = np.sin(colatr) # st = "sine theta" (theta = colatitude)
    sp = np.sin(lonr)   # sp = "sine phi"   (phi   = longitude )
    
    ct = np.cos(colatr) 
    cp = np.cos(lonr)

    e =  np.vstack([-sp   ,     cp, sp*0]).T
    n = -np.vstack([-ct*cp, -ct*sp,   st]).T
    u =  np.vstack([ st*cp,  st*sp,   ct]).T

    # if e.shape[0] == 1:
    #     e = e.ravel()
    #     n = n.ravel()
    #     u = u.ravel()

    return e,n,u


def get_receiver_az_el_geod_ECEF(gdlatRec,glonRec,R_S,hRec=0.):
    """
    Given the location of the receiver (as a gdlat,glon pair) and the 
    location of the scattering point R_S (as an ECEF point), 
    calculate and return the receiver's azimuth and elevation in degrees.

    hRec : float
            Geodetic altitude of the receiver in km

    Spencer Mark Hatch
    2022/03/22
    """

    assert (R_S.size == 3) 
    if R_S.shape != (1,3):
        R_S = R_S[np.newaxis,:]

    # # local enu vectors at transceiver
    # e_enu = np.array([1,0,0])
    # n_enu = np.array([0,1,0])
    # u_enu = np.array([0,0,1])

    # theta, r, n_theta, n_r = geod2geoc(gdlatRec, hRec, n_enu[1], -n_enu[2])
    # theta, r, u_theta, u_r = geod2geoc(gdlatRec, hRec, u_enu[1], -u_enu[2])
    
    theta, r, _, _ = geod2geoc(gdlatRec, hRec, 0., 0.)
    theta, r, _, _ = geod2geoc(gdlatRec, hRec, 0., 0.)

    rRec, gclatRec = r, 90.-theta

    # # geodetic e,n,u vectors with spherical enu (phi,-theta,r) components
    # e_sphenu = np.array([1,0,0])
    # n_sphenu = np.array([0,-n_theta,n_r])
    # u_sphenu = np.array([0,-u_theta,u_r])

    # Rvec_SPHENU_TO_ECEF = ECEFtosphericalENUMatrix(glonRec, gclatRec).squeeze().T

    # # geodetic e,n,u vectors with Cartesian ECEF components
    # eR = (Rvec_SPHENU_TO_ECEF@e_sphenu)[:,np.newaxis].T
    # nR = (Rvec_SPHENU_TO_ECEF@n_sphenu)[:,np.newaxis].T
    # uR = (Rvec_SPHENU_TO_ECEF@u_sphenu)[:,np.newaxis].T

    eR, nR, uR = get_geodetic_enu_vectors_in_ECEF(gdlatRec, glonRec, hRec=hRec)

    thr = np.deg2rad(theta)
    phr = np.deg2rad(glonRec)
    sth, cth = np.sin(thr), np.cos(thr)
    sph, cph = np.sin(phr), np.cos(phr)

    # NOTE: Here we want to use the geocentric radial vector (in ECEF coords), not geodetic "up" vector!
    # This 
    R_R = rRec * np.array([sth*cph, sth*sph, cth])[np.newaxis,:]

    dR = R_S-R_R

    # Get receiver azimuth for pointing at R_S
    azR = np.arctan2(dotprod(dR,eR),dotprod(dR,nR))

    # Azimuthal vector
    e_azR = np.sin(azR)*eR + np.cos(azR) * nR

    # Get receiver elevation for pointing at R_S
    elR = np.arctan2(dotprod(dR,uR),dotprod(dR,e_azR))

    return np.rad2deg(azR).squeeze(), np.rad2deg(elR).squeeze()


def get_receiver_az_el_geod(gdlatRec,glonRec,gdlatScat,glonScat,hScat,hRec=0.):
    """
    Given the location of the receiver (as gdlatRec,glonRec, and possibly hRec) and the 
    location of the scattering point (as gdlatScat,glonScat, and hScat), 
    calculate and return the receiver's azimuth and elevation in degrees.

    Spencer Mark Hatch
    2022/03/22
    """

    thetaScat, rScat, _, _ = geod2geoc(gdlatScat, hScat, 0., 0.,)

    R_S = rScat * rvec(thetaScat, glonScat)

    return get_receiver_az_el_geod_ECEF(gdlatRec,glonRec,R_S,hRec=hRec)


# def get_range_line(gclat, gclon, az, el, h_km, degrees=True,returnbonus=False):

#     warnings.warn("This thing assumes geocentric coordinates are right! In particular, it assume that the geocentric radial unit vector is the same as the geodetic 'up' vector, which is of course not totally true!")

#     # assert 2<0,"Where is this being used? We should standardize and make everything use geodetic coordinates"

#     e, n, u = get_enu_vectors_cartesian(gclat,gclon,degrees=degrees)

#     R = u * geodeticheight2geocentricR(geocentric2geodeticlat(gclat), 0.)
#     # R = RE * u

#     phat = point_vector_cartesian(gclat,gclon,az,el,degrees=degrees)
#     # px, py, pz = point_vector_cartesian(gclat,gclon,az,el,degrees=degrees)

#     d = height_to_range(h_km, el, degrees=degrees)

#     r_los = R + d[...,np.newaxis] * phat

#     if returnbonus:
#         return r_los,dict(d=d,phat=phat,R=R,e=e,n=n,u=u)
#     else:
#         return r_los


def get_range_line(gdlatRec, glonRec, az, el, h_km, hRec=0., degrees=True, returnbonus=False):

    eR, nR, uR = get_geodetic_enu_vectors_in_ECEF(gdlatRec, glonRec, hRec=hRec)
    
    thetaRec, rRec, _, _ = geod2geoc(gdlatRec, hRec, 0., 0.)

    # Vector pointing to receiver from Earth center in ECEF (Cartesian) coordinates
    R = rRec * rvec(thetaRec, glonRec)
    
    azr, elr = np.deg2rad(az), np.deg2rad(el)

    azhat = np.sin(azr)[:,np.newaxis] * eR + np.cos(azr)[:,np.newaxis] * nR  # unit vector in azimuth direction

    phat = np.sin(elr)[:,np.newaxis] * uR + np.cos(elr)[:,np.newaxis] * azhat

    # phat = point_vector_cartesian(gclat,gclon,az,el,degrees=degrees)
    # px, py, pz = point_vector_cartesian(gclat,gclon,az,el,degrees=degrees)

    d = height_to_range(h_km, el, degrees=degrees)

    r_los = R + d[...,np.newaxis] * phat

    if returnbonus:
        return r_los,dict(d=d,phat=phat,R=R,e=e,n=n,u=u)
    else:
        return r_los


def ECEFtosphericalENUMatrix(lon, gclat):
    """

    First  row is "East"  (phi)      vector (   phihat = -sinp xhat + cosp yhat)
    Second row is "North" (90-theta) vector (lambdahat = -sinl cosp xhat - cosl sinp yhat + cosl zhat)
    Third  row is "Up"    (radial)   vector (     rhat =  cosl cosp xhat + cosl sinp yhat + sinl zhat)

    Here phi    ("p") is the azimuthal coordinate in spherical coordinates
         lambda ("l") is the 90°-theta latitude angle


    Naturally, the columns of this matrix give xhat, yhat, and zhat (in that order) in terms of 
    ehat, nhat, and uhat

    So! Multiply a vector with ENU components by this matrix to get back a vector in ECEF coordinates
    """

    phi = np.deg2rad(lon)
    lamb = np.deg2rad(gclat)

    return np.stack([np.vstack([             -np.sin(phi),               np.cos(phi),  np.zeros(phi.size)]),
                     np.vstack([-np.sin(lamb)*np.cos(phi), -np.sin(lamb)*np.sin(phi),        np.cos(lamb)]),
                     np.vstack([ np.cos(lamb)*np.cos(phi),  np.cos(lamb)*np.sin(phi),        np.sin(lamb)])])


def rvec(thetad,phid):
        
    thr = np.deg2rad(thetad)
    phr = np.deg2rad(phid)
    sth, cth = np.sin(thr), np.cos(thr)
    sph, cph = np.sin(phr), np.cos(phr)

    return np.array([sth*cph, sth*sph, cth])[np.newaxis,:]


# if __name__ == "__main__":

#     # Skibotn
#     SKI = (69.58333, 20.46667)
#     SKI = (69.3401035, 20.315087)  # Anders's suggestion from Google Maps

    
#     KAI = (68.29,19.45)        # Kaiseniemi
#     KRS = (68.44933, 22.48325) # Karesuvanto

    
#     TX = SKI


#     el_arr1=np.array([64, 61, 60, 58, 57, 55, 54, 54, 57, 59, 61, 61])
#     az_arr1=np.array([0, 35, 69, 101, 130, 156, 180, 204, 231, 258,288, 323])
#     el_arr2=np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30])
#     az_arr2=np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270,300, 330])
#     el_arr3=np.array([66, 77.8, 90])
#     az_arr3=np.array([180, 180, 180])

#     # az_arr = az_arr1 + az_arr2 + az_arr3
#     # el_arr = el_arr1 + el_arr2 + el_arr3
#     az_arr = np.concatenate((az_arr1,az_arr2,az_arr3))
#     el_arr = np.concatenate((el_arr1,el_arr2,el_arr3))

#     h_km = 100

#     r_los,d = get_range_line(*TX, az_arr1, el_arr1, h_km,returnbonus=True)
#     d,phat,R,e,n,u = d['d'], d['phat'], d['R'], d['e'],d['n'],d['u']

