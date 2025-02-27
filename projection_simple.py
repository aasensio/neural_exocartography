import numpy as np
import healpy as hp


def logit(x, low=0, high=1):
    r"""Returns the logit function at ``x``.  Logit is defined as

    .. math::

      \mathrm{logit}\, x \equiv \log(x - \mathrm{low}) - \log(\mathrm{high} - x)

    """
    return np.log(x-low) - np.log(high-x)

def inv_logit(y, low=0, high=1):
    """Returns the ``x`` such that ``y == logit(x, low, high)``.

    """
    ey = np.exp(y)
    return low/(1.0 + ey) + high/(1.0 + 1.0/ey)

def quaternion_multiply(qa, qb):
    result = np.zeros(np.broadcast(qa, qb).shape)
    
    result[..., 0] = qa[..., 0]*qb[..., 0] - np.sum(qa[..., 1:]*qb[..., 1:], axis=-1)
    result[..., 1] = qa[..., 0]*qb[..., 1] + qa[..., 1]*qb[..., 0] + qa[..., 2]*qb[..., 3] - qa[..., 3]*qb[..., 2]
    result[..., 2] = qa[..., 0]*qb[..., 2] - qa[..., 1]*qb[..., 3] + qa[..., 2]*qb[..., 0] + qa[..., 3]*qb[..., 1]
    result[..., 3] = qa[..., 0]*qb[..., 3] + qa[..., 1]*qb[..., 2] - qa[..., 2]*qb[..., 1] + qa[..., 3]*qb[...,0]

    return result

def rotation_quaternions(axis, angles):
    angles = np.atleast_1d(angles)
    result = np.zeros((angles.shape[0], 4))
    result[:, 0] = np.cos(angles/2.0)
    result[:, 1:] = np.sin(angles/2.0)[:, np.newaxis]*axis

    return result

def rotate_vector(rqs, v):
    nrs = rqs.shape[0]

    rqs = rqs

    vq = np.zeros((nrs, 4))
    vq[:,1:] = v

    result = quaternion_multiply(rqs, vq)
    rqs[:,1:] *= -1
    result = quaternion_multiply(result, rqs)

    return result[:,1:]

class IlluminationMapPosterior(object):
    """A posterior class for mapping surfaces from a reflectance time series"

    :param times:
        Array of times of photometric measurements.

    :param reflectance:
        Array of photometric measurements at corresponding `times`.

    :param sigma_reflectance:
        Single value or array of 1-sigma uncertainties on reflectance measurements.

    :param nside: (optional)
        Resolution of HEALPix surface map.  Has to be a power of 2.  The number
        of pixels will be :math:`12 N_\mathrm{side}^2`.
        (default: ``4``)

    :param nside_illum: (optional):
        Resolution of HEALPix illumination map, i.e., the "kernel" of
        illumination that is integrated against the pixel map.
        (default: ``16``)

    :param map_parameterization: (optional):
        Parameterization of surface map to use.  `pix` will parameterize the map with
        values of each pixel; `alm` will parameterize the map in spherical harmonic coefficients.
        (default: ``pix``)

    """
    def __init__(self, times, reflectance, sigma_reflectance, nside=4, nside_illum=16, map_parameterization='pix'):
        assert nside_illum >= nside, 'IlluminationMapPosterior: must have nside_illum >= nside'

        self._map_parameterization = map_parameterization

        self._times = np.array(times)
        self._reflectance = np.array(reflectance)
        self._sigma_reflectance = sigma_reflectance * np.ones_like(times)
        self._nside = nside
        self._nside_illum = nside_illum

        self._npix_illum = hp.nside2npix(self._nside_illum)

        self._fixed_params = {}

        self.pts = hp.pix2vec(self._nside_illum, np.arange(0, self._npix_illum))
        self.pts = np.column_stack(self.pts)

        self.area = hp.nside2pixarea(self._nside_illum)
    
    def visibility_illumination_matrix(self, p):
        
        phi_orb = inv_logit(p['logit_phi_orb'], 0.0, 2.0*np.pi)
        cos_phi_orb = np.cos(phi_orb)
        sin_phi_orb = np.sin(phi_orb)
        
        omega_orb = 2.0*np.pi / np.exp(p['log_orbital_period'])

        no = self._observer_normal_orbit_coords(p)
        ns = np.array([-cos_phi_orb, -sin_phi_orb, 0.0])

        orb_quats = rotation_quaternions(np.array([0.0, 0.0, 1.0]), -omega_orb*self._times)        

        to_body_frame_quats = self._body_frame_quaternions(p, self._times)

        star_to_bf_quats = quaternion_multiply(to_body_frame_quats,orb_quats)        

        nos = rotate_vector(to_body_frame_quats, no)
        nss = rotate_vector(star_to_bf_quats, ns)        

        cos_insolation = np.sum(nss[:,np.newaxis,:]*self.pts[np.newaxis,:,:], axis=2)
        cos_insolation[cos_insolation > 1] = 1.0
        cos_insolation[cos_insolation < 0] = 0.0

        cos_obs = np.sum(nos[:,np.newaxis,:]*self.pts[np.newaxis,:,:], axis=2)
        cos_obs[cos_obs > 1] = 1.0
        cos_obs[cos_obs < 0] = 0.0

        cos_factors = cos_insolation*cos_obs        

        return self.area * cos_factors/np.pi

    def _body_frame_quaternions(self, p, times):
        
        cos_obl = inv_logit(p['logit_cos_obl'], 0.0, 1.0)
        sin_obl = np.sqrt(1.0 - cos_obl*cos_obl)
        obl = np.arccos(cos_obl)

        obl_orientation = inv_logit(p['logit_obl_orientation'], 0.0, 2.0*np.pi)
        cos_obl_orientation = np.cos(obl_orientation)
        sin_obl_orientation = np.sin(obl_orientation)

        omega_rot = 2.0*np.pi/np.exp(p['log_rotation_period'])

        S = np.array([cos_obl_orientation*sin_obl, sin_obl_orientation*sin_obl, cos_obl])

        spin_rot_quat = quaternion_multiply(rotation_quaternions(np.array([0.0, 1.0, 0.0]), -obl), \
                                            rotation_quaternions(np.array([0.0, 0.0, 1.0]), -obl_orientation))

        rot_quats = rotation_quaternions(S, -omega_rot*self._times)

        return quaternion_multiply(spin_rot_quat,rot_quats)

    def _observer_normal_orbit_coords(self, p):
        
        cos_inc = inv_logit(p['logit_cos_inc'], 0.0, 1.0)
        sin_inc = np.sqrt(1.0 - cos_inc*cos_inc)

        return np.array([-sin_inc, 0.0, cos_inc])

    
    def lightcurve(self, p, map):
        
        V = self.visibility_illumination_matrix(p)
        
        map_lc = np.dot(V, map)

        return map_lc

