import numpy as np
from tqdm import tqdm
import scipy.sparse.linalg

def prox_l2(x, thr):
    return np.fmax(1.0 - thr / np.linalg.norm(x), 0.0) * x

def simplex_noise(x, y, z, freq, weight):
    return weight * (noise.noise3d(x*freq, y*freq, z*freq) / 2.0 + 0.5)

def tikhonov(Phi, y, theta=0.01, iter=2000):

    largest_eval = scipy.sparse.linalg.eigsh(Phi.T @ Phi, k=1, which='LM', return_eigenvectors=False)
    rho = 0.4 / largest_eval

    d = Phi.T @ y
        
    x = Phi.T @ y #np.zeros_like(d)
    pbar = tqdm(range(iter))
    loop = 0
    for i in pbar:
        t = Phi @ x
        residual = Phi.T @ t - d
        r = x - rho * residual
        x = prox_l2(r, theta)

        pbar.set_postfix(iter=loop, loss=np.linalg.norm(residual))

        loop += 1

    return x

if (__name__ == '__main__'):

    import healpy as hp
    from opensimplex import OpenSimplex
    import exocartographer
    import matplotlib.pyplot as pl

    pl.close('all')

    seed = np.random.randint(low=0, high=10000)
    noise = OpenSimplex(seed=125)#seed=123)

    # Set orbital properties
    p_rotation = 23.934
    p_orbit = 365.256363 * 24.0
    phi_orb = np.pi
    inclination = 0.0 #np.pi/2
    obliquity = 90. * np.pi/180.0
    phi_rot = np.pi

    nside = 16
    npix = hp.nside2npix(nside)
    polar_angle, azimuthal_angle = hp.pixelfunc.pix2ang(nside, np.arange(npix))

    x = np.sin(polar_angle) * np.cos(azimuthal_angle)
    y = np.sin(polar_angle) * np.sin(azimuthal_angle)
    z = np.cos(polar_angle)

    simulated_map = np.zeros(npix)
    for i in range(npix):
        simulated_map[i] = simplex_noise(x[i], y[i], z[i], 1.0, 1.0) + simplex_noise(x[i], y[i], z[i], 2.0, 1/4.0) + simplex_noise(x[i], y[i], z[i], 4.0, 1/16.0)
    
    simulated_map = simulated_map**2.5
    thr = 0.3
    simulated_map[simulated_map < thr] = 0.0
    max_tmp = np.max(simulated_map)
    tmp = simulated_map >= thr
    simulated_map[tmp] -= thr
    simulated_map[tmp] /= max_tmp - thr

    # Observation schedule
    cadence = p_rotation/24.
    nobs_per_epoch = 24
    epoch_duration = nobs_per_epoch * cadence

    epoch_starts = [30*p_rotation, 60*p_rotation, 150*p_rotation,
                    210*p_rotation, 250*p_rotation]

    times = np.array([])
    for epoch_start in epoch_starts:
        epoch_times = np.linspace(epoch_start,
                                epoch_start + epoch_duration,
                                nobs_per_epoch)
        times = np.concatenate([times, epoch_times])

    measurement_std = 0.001

    truth = exocartographer.IlluminationMapPosterior(times, np.zeros_like(times),
                                    measurement_std, nside=nside, nside_illum=nside)

    true_params = {
        'log_orbital_period':np.log(p_orbit),
        'log_rotation_period':np.log(p_rotation),
        'logit_cos_inc':exocartographer.util.logit(np.cos(inclination)),
        'logit_cos_obl':exocartographer.util.logit(np.cos(obliquity)),
        'logit_phi_orb':exocartographer.util.logit(phi_orb, low=0, high=2*np.pi),
        'logit_obl_orientation':exocartographer.util.logit(phi_rot, low=0, high=2*np.pi)}
    truth.fix_params(true_params)
    p = np.concatenate([np.zeros(truth.nparams), simulated_map])

    Phi = truth.visibility_illumination_matrix(p)

    light = truth.lightcurve(p)
                
    y = Phi @ simulated_map

    rho = 10.0
    theta = 0.01
    
    x = tikhonov(Phi, y, theta)

    hp.mollview(x)
    hp.mollview(simulated_map)

    pl.figure()
    pl.plot(light)    
    pl.plot(Phi @ x)

    pl.show()