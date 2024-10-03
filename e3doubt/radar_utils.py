import numpy as np
import ppigrf
from datetime import datetime
from .vectors import dotprod,crossprod,vecmag
from .geodesy import geodeticheight2geocentricR,geocentric2geodeticlat,geod2geoc,geoc2geod,ECEF2geodetic,geodetic2geocentricXYZ

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


def height_to_range(h_km,el,gdlat=None,degrees=True):
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

    if gdlat is None:
        R = RE
    else:
        R = geodeticheight2geocentricR(gdlat,0.)

    return np.sqrt(R**2 * np.sin(elr)**2 + 2*R*h_km + h_km**2) - R*np.sin(elr)


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


def get_geocentric_enu_vectors_cartesian(gclat,gclon,degrees=True):
    """
    Get geocentric east, north, up vectors for a given (lat,lon) pair or a list 
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

    e =  np.vstack([-sp   ,     cp, sp*0]).T
    n = -np.vstack([-ct*cp, -ct*sp,   st]).T
    u =  np.vstack([ st*cp,  st*sp,   ct]).T

    # if e.shape[0] == 1:
    #     e = e.ravel()
    #     n = n.ravel()
    #     u = u.ravel()

    return e,n,u


def get_points_az_el_geod_ECEF(gdlatRec,glonRec,R_S,hRec=0.):
    """
    Given the location of the receiver (as a gdlat,glon pair) and the 
    locations of the scattering point R_S (as ECEF points), 
    calculate and return the receiver's azimuth and elevation in degrees.

    NOTE: This is just a version of get_point_az_el_geod_ECEF when input R_S is an array

    hRec : float
            Geodetic altitude of the receiver in km

    Spencer Mark Hatch
    2024/09/20
    """

    # 
    if len(R_S.shape) == 1:
      assert (R_S.size == 3)   
      if R_S.shape != (1,3):
          R_S = R_S[np.newaxis,:]

    elif len(R_S.shape) == 2:
        assert R_S.shape[1] == 3,"R_S must have shape (N, 3) or (3)"

        # N = R_S.shape[0]
        # print(f"Getting az, el for {N} points")

    # # local enu vectors at transceiver
    # e_enu = np.array([1,0,0])
    # n_enu = np.array([0,1,0])
    # u_enu = np.array([0,0,1])

    theta, r, _, _ = geod2geoc(gdlatRec, hRec, 0., 0.)
    theta, r, _, _ = geod2geoc(gdlatRec, hRec, 0., 0.)

    rRec, gclatRec = r, 90.-theta

    eR, nR, uR = get_geodetic_enu_vectors_in_ECEF(gdlatRec, glonRec, hRec=hRec)

    thr = np.deg2rad(theta)
    phr = np.deg2rad(glonRec)
    sth, cth = np.sin(thr), np.cos(thr)
    sph, cph = np.sin(phr), np.cos(phr)

    # NOTE: Here we want to use the geocentric radial vector (in ECEF coords), not geodetic "up" vector!
    R_R = rRec * np.array([sth*cph, sth*sph, cth])[np.newaxis,:]

    dR = R_S-R_R

    # Get receiver azimuth for pointing at R_S
    azR = np.arctan2(dotprod(dR,eR),dotprod(dR,nR))

    # Azimuthal vector
    e_azR = (np.sin(azR)*np.broadcast_to(eR,(len(azR),3)).T).T + (np.cos(azR)*np.broadcast_to(nR,(len(azR),3)).T).T

    # Get receiver elevation for pointing at R_S
    elR = np.arctan2(dotprod(dR,uR),dotprod(dR,e_azR))

    return np.rad2deg(azR).squeeze(), np.rad2deg(elR).squeeze()



def get_point_az_el_geod_ECEF(gdlatRec,glonRec,R_S,hRec=0.):
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


def get_point_az_el_geod(gdlatRec,glonRec,gdlatScat,glonScat,hScat,hRec=0.):
    """
    Given the location of the receiver (as gdlatRec,glonRec, and possibly hRec) and the 
    location of the scattering point (as gdlatScat,glonScat, and hScat), 
    calculate and return the receiver's azimuth and elevation in degrees.

    Spencer Mark Hatch
    2022/03/22
    """

    thetaScat, rScat, _, _ = geod2geoc(gdlatScat, hScat, 0., 0.,)

    R_S = rScat * rvec(thetaScat, glonScat)

    return get_point_az_el_geod_ECEF(gdlatRec,glonRec,R_S,hRec=hRec)


def get_2D_csgrid_az_el(gdlat_t, glon_t, h_grid=200, L=100e3, W=100e3, Lres=10e3, Wres=10e3,
                        wshift = None,
                        return_grid=False,
                        ctr_on_fieldaligned_beam=False,
                        # ctr__refheight=200.,
):
    """
    Get azimuths and elevations of selection of points defined on a 2D grid in cubed sphere coordinates.

    Inputs
    ======
    gdlat_t : float, default False
        Geodetic latitude of the transmitter site in deg
    glon_t  : float, default False
        Geographic longitude of the transmitter site in deg


    Keywords
    ========
    h_grid  : float, default 200.
        Reference altitude of the grid in km
    L       : float, default 100e3
        Length of the grid in meters(NB!)
    W       : float, default 100e3
        Width of the grid in meters(NB!)
    Lres    : float, default 10e3
        Grid spacing along the length of the grid in meters
    Wres    : float, default 10e3
        Grid spacing along the width of the grid in meters
    wshift  : float, default None
        Some shift along the width direction in meters?

    ctr_on_fieldaligned_beam  : bool, default False
        If True, center the grid around the transmitter beam (located at gdlat_t, glon_t)
        that is most aligned with Earth's magnetic field. The geodlat and glon of this 
        location is found using the function 'get_field_aligned_beam' and reference height 
        'h_grid'.

    Outputs
    =======

    if return_grid:
        return az, el, gdlat, glon, h, (grid, projection)
    else:
        return az, el, gdlat, glon, h
    """
    
    try:
        from secsy import cubedsphere as cs
    
    except:
        print("Couldn't import cubedsphere! Try install secsy package")
        return 0

    # Convert most inputs to meters
    # L = L*1e3
    # W = W*1e3
    # Lres = Lres*1e3
    # Wres = Wres*1e3

    if ctr_on_fieldaligned_beam:
        print(f"Finding gdlat/glon of field-aligned beam at ref height={h_grid} km")
        az1,el1, gdlat1, glon1 = get_field_aligned_beam(h_grid,
                                                        gdlat_tx=gdlat_t,
                                                        glon_tx=glon_t,
                                                        ddeg=0.02,
                                                        degdimlat=5.0,
                                                        degdimlon=8.0,
        )
        ctrgdlat, ctrglon = gdlat1, glon1
    else:
        ctrgdlat, ctrglon = gdlat_t, glon_t

    if wshift is None:
        wshift = -Wres//2

    # theta, RI, _, _ = geod2geoc(gdlat_t, h_grid, 0., 0.,)
    # gclat_t = 90.-theta
    theta, RI, _, _ = geod2geoc(ctrgdlat, h_grid, 0., 0.,)
    ctrgclat = 90.-theta
    RI *= 1e3                   # to meters
    
    # projection = cs.CSprojection((glon_t, gclat_t), 0) 
    projection = cs.CSprojection((ctrglon, ctrgclat), 0) 
    grid = cs.CSgrid(projection, L, W, Lres, Wres, R = RI, wshift = wshift)
    
    gclat, glon = grid.lat.ravel(), grid.lon.ravel()
    gdlat, h, _, _ = geoc2geod(90.-gclat, grid.R/1e3, 0., 0.)
    assert np.all(np.isclose(h,h_grid,atol=0.2))  # permit inaccurancy of 200 m
    npt = gdlat.size
    
    az, el = np.zeros(npt), np.zeros(npt)
    
    for i,(gdla, glo, hi) in enumerate(zip(gdlat,glon,h)):
        # print(i,gdla,glo)
        
        azt, elt = get_point_az_el_geod(gdlat_t,glon_t,gdla,glo,hi,hRec=0.)
    
        az[i] = azt
        el[i] = elt
        

    if return_grid:
        return az,el,gdlat,glon,h,grid
    else:
        return az,el,gdlat,glon,h


def get_range_line(gdlatRec, glonRec, az, el, h_km, hRec=0., returnbonus=False, brent=False):
    """
    r_los = get_range_line(gdlatRec, glonRec, az, el, h_km)

    Given radar lat, lon, azimuth, elevation, altitude of a point in question (and optionally radar altitude), 
    return a vector in ECEF coordinates that points from the center of the earth to the point along the radar line of sight.

    gdlatRec: Radar geodetic latitude                   [deg]
    glonRec : Radar geographic longitude                [deg]
    az      : Radar azimuth                             [deg]
    el      : Radar elevation                           [deg]
    h_km    : Geodetic altitude of point(s) in question [km]
    hRec    : Geodetic altitude of radar                [km]

    return_bonus : If true, also return dictionary containing range (d), 
                   unit vector pointing from radar to point in question (phat) in ECEF coordinates, 
                   vector pointing from center of Earth to radar (R) in ECEF coordinates, 
                   and east, north, and up unit vectors (e, n, u) at radar in ECEF coordinates.

    brent   : Use Brent's method to calculate range (requires scipy.optimize)
    NOTE
    =====
    If brent=False, the function height_to_range is used. This assumes that the Earth is spherical,
    and incurs a small error of a few % for altitudes of a couple hundred kilometers.
    """

    eR, nR, uR = get_geodetic_enu_vectors_in_ECEF(gdlatRec, glonRec, hRec=hRec)
    
    thetaRec, rRec, _, _ = geod2geoc(gdlatRec, hRec, 0., 0.)

    # Vector pointing to receiver from Earth center in ECEF (Cartesian) coordinates
    R = rRec * rvec(thetaRec, glonRec)
    
    azr, elr = np.deg2rad(az), np.deg2rad(el)

    azhat = np.sin(azr)[:,np.newaxis] * eR + np.cos(azr)[:,np.newaxis] * nR  # unit vector in azimuth direction

    phat = np.sin(elr)[:,np.newaxis] * uR + np.cos(elr)[:,np.newaxis] * azhat

    # phat = point_vector_cartesian(gclat,gclon,az,el,degrees=degrees)
    # px, py, pz = point_vector_cartesian(gclat,gclon,az,el,degrees=degrees)

    if not brent:
        d = height_to_range(h_km, el, gdlat=gdlatRec, degrees=True)
    else:
        from scipy.optimize import brent
        
        d = np.zeros_like(el)
        for i in range(len(el)):
            # aztmp = azr[i]
            eltmp = el[i]
            azhtmp = azhat[i,:]
            phtmp = phat[i,:]
            htmp = h_km[i]
            dest = height_to_range(htmp, eltmp, gdlat=gdlatRec, degrees=True)

            f = lambda d: np.abs(ECEF2geodetic(*(R+d*phtmp).T,degrees=True)[0][0]-htmp)
            d[i] = brent(f, brack=(0., dest, dest*2))

    r_los = R + d[...,np.newaxis] * phat

    if returnbonus:
        return r_los,dict(d=d,phat=phat,R=R,e=eR,n=nR,u=uR)
    else:
        return r_los


def get_field_aligned_beam(h,gdlat_tx=69.34,glon_tx=20.31,
                           ddeg=0.025,
                           degdimlat=5.0,
                           degdimlon=10.0,
                           igrf_refdate = datetime(2020,12,1),
):
    """
    Inputs
    =====
    h        : float, default 200.
        Geodetic altitude in km of sphere over which to find most field-aligned az,el

    Keywords
    ========
    gdlat_tx : float, default Skibotn transceiver geod lat
        Geodetic latitude of transmitter at h = 0 km
    glon_tx  : float, default Skibotn transceiver geod lat
        Geographic longitude of transmitter at h = 0 km
    ddeg     : float, default 0.05
        Spacing between grid points in latitude (and longitude)
    degdimlat: float, default 10.0
        Span of grid around gdlat_tx, (gdlat_tx±degdimlat)
    degdimlon: float, default 20.0
        Span of grid around glon_tx, (glon_tx±degdimlat)

    Outputs
    ========
    az,el,gdlat,glon : azimuth, elevation, geodetic latitude, and geographic longitude of most field-aligned point
    """

    dlat = np.arange(-degdimlat,degdimlat,ddeg)
    dlon = np.arange(-degdimlon,degdimlon,ddeg)
    
    dlat, dlon = np.meshgrid(dlat,dlon,indexing='ij')
    
    dlshape = dlat.shape
    
    dlat, dlon = dlat.ravel(), dlon.ravel()
    
    gdlatp = gdlat_tx + dlat
    glonp = glon_tx+dlon
    
    xt, yt, zt = geodetic2geocentricXYZ(gdlat_tx, glon_tx, 0, returnR = False, degrees = True)
    xp, yp, zp = geodetic2geocentricXYZ(gdlatp, glonp, h, returnR = False, degrees = True)
    
    thetap, rp, _, _ = geod2geoc(gdlatp, h, 0., 0.)
    gclatp = 90.-thetap
    
    ## 2. Get B-field unit vectors b_k at points in question
    
    Bp_sph = np.vstack(ppigrf.igrf_gc(rp,thetap,glonp,igrf_refdate))
    magBp = np.linalg.norm(Bp_sph,axis=0)
    
    # unit B-field vectors in spherical (r, theta, phi) coordinates
    bp_sph = Bp_sph/np.linalg.norm(Bp_sph,axis=0)
    bp_sphenu = np.stack([bp_sph[2], -bp_sph[1], bp_sph[0]])
    
    Rvec_SPHENU_TO_ECEF_p = ECEFtosphericalENUMatrix(glonp, gclatp)
    
    bp_ecef = np.einsum('ij...,i...',Rvec_SPHENU_TO_ECEF_p,bp_sphenu).T
    
    ## 3. Get unit vectors l_k that point from points in question to Skibotn
    L_ecef = np.stack([xp-xt,yp-yt,zp-zt])
    l_ecef = -L_ecef/np.linalg.norm(L_ecef,axis=0)
    
    ## 4. Find point k for which dot(b_k,l_k) is maximized (or actually minimized)
    dot_bl = np.sum(l_ecef*bp_ecef,axis=0)
    
    win_ip = np.argmax(dot_bl)
    
    angler = np.rad2deg(np.arccos(dot_bl))
    # angler[angler > 90] = 90-angler
    angler = angler.reshape(dlshape)
    
    gdlat, glon = gdlatp[win_ip],glonp[win_ip]

    ## 5. Get el, az of this point
    az, el = get_point_az_el_geod(gdlat_tx,glon_tx,gdlatp[win_ip],glonp[win_ip],h)
        
    return az,el,gdlat,glon


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
    """
    Get spherical r unit vector in ECEF coordinates given colatitude and azimuth thetad and phid (in degrees)
    """
        
    thr = np.deg2rad(thetad)
    phr = np.deg2rad(phid)
    sth, cth = np.sin(thr), np.cos(thr)
    sph, cph = np.sin(phr), np.cos(phr)

    return np.array([sth*cph, sth*sph, cth])[np.newaxis,:]


def get_kvec(x1,y1,z1,x2,y2,z2):
    """
    Get unit vector(s) that point from location 1 (given by x1, y1, z1) to location 2 (x2, y2, z2) for Cartesian coordinates
    """
    # points from 1 to 2
    dx12, dy12, dz12 = x2-x1,y2-y1,z2-z1

    #get unit vector
    khat = np.vstack([dx12,dy12,dz12])
    khat = khat/np.linalg.norm(khat,axis=0)

    return khat


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

