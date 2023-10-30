import numpy as np
from vectors import dotprod,crossprod,vecmag
from geodesy import geodeticheight2geocentricR,geocentric2geodeticlat
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


def get_enu_vectors_cartesian(gclat,gclon,degrees=True):
    """
    Get east, north, up vectors for a given (lat,lon) pair or a list 
    of (lat,lon) pairs in Cartesian (Earth-centered, Earth-fixed, or 
    ECEF) coordinates.

    Spencer Mark Hatch
    2022/03/22
    """
    
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

    e =  np.vstack([-sp   ,    cp, sp*0]).T
    n = -np.vstack([-ct*cp, ct*sp,  -st]).T
    u =  np.vstack([ st*cp, st*sp,   ct]).T

    # if e.shape[0] == 1:
    #     e = e.ravel()
    #     n = n.ravel()
    #     u = u.ravel()

    return e,n,u


def get_receiver_az_el(gclatRec,gclonRec,R_S,degrees=True):
    """
    Given the location of the receiver (as a lat,lon pair) and the 
    location of the scattering point R_S (as an ECEF point), 
    calculate the receiver's azimuth and elevation.

    Spencer Mark Hatch
    2022/03/22
    """

    warnings.warn("This code is not tested!")

    eR, nR, uR = get_enu_vectors_cartesian(gclatRec,gclonRec,degrees=degrees)

    R_R = RE * uR

    dR = R_S-R_R

    # Get receiver azimuth for pointing at R_S
    azR = np.arctan2(dotprod(dR,eR),dotprod(dR,nR))

    # Azimuthal vector
    e_azR = np.sin(azR)*eR + np.cos(azR) * nR

    # Get receiver elevation for pointing at R_S
    elR = np.arctan2(dotprod(dR,uR),dotprod(dR,e_azR))

    return azR, elR


def get_range_line(gclat, gclon, az, el, h_km, degrees=True,returnbonus=False):

    e, n, u = get_enu_vectors_cartesian(gclat,gclon,degrees=degrees)

    R = u * geodeticheight2geocentricR(geocentric2geodeticlat(gclat), 0.)
    # R = RE * u

    phat = point_vector_cartesian(gclat,gclon,az,el,degrees=degrees)
    # px, py, pz = point_vector_cartesian(gclat,gclon,az,el,degrees=degrees)

    d = height_to_range(h_km, el, degrees=degrees)

    r_los = R + d[...,np.newaxis] * phat

    if returnbonus:
        return r_los,dict(d=d,phat=phat,R=R,e=e,n=n,u=u)
    else:
        return r_los


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




