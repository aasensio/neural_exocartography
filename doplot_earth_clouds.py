import numpy as np
import matplotlib.pyplot as pl
import torch
import model_denoise_clouds as model
import glob
import os
import exocartographer
import healpy as hp
import tikhonov
import pandas as pd
import time
import scipy.sparse.linalg
from PIL import Image
import subprocess
from tqdm import tqdm
import datetime
import cartopy.feature as cfeature

def simplex_noise(noise, x, y, z, freq, weight):
    return weight * (noise.noise3d(x*freq, y*freq, z*freq) / 2.0 + 0.5)

def billow_noise(noise, x, y, z, freq, weight):
    return weight * (2.0 * np.abs(noise.noise3d(x*freq, y*freq, z*freq)) - 1.0)


def weighted_avg_and_std(values, weights, axis=0):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=axis)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return average, np.sqrt(variance)

class Testing(object):
    def __init__(self, gpu=0, checkpoint_1d=None, checkpoint_2d=None, K1d=3, K2d=3, model_type='sparse'):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        self.K1d = K1d
        self.K2d = K2d
        self.model_type = model_type

        self.checkpoint_1d = checkpoint_1d
        self.checkpoint_2d = checkpoint_2d
                
        if (self.model_type == 'denoise'):
            self.model_1d = model.Network(K=self.K1d, L=32, device=self.device, model_class='conv1d').to(self.device)        
            print('N. total parameters 1D : {0}'.format(sum(p.numel() for p in self.model_1d.parameters() if p.requires_grad)))
            print("=> loading checkpoint '{}'".format(self.checkpoint_1d))
            checkpoint = torch.load(self.checkpoint_1d, map_location=lambda storage, loc: storage)
            self.model_1d.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(self.checkpoint_1d))    
            print(f"rho : {checkpoint['state_dict']['rho']}")

            self.model_2d = model.Network(K=self.K2d, L=32, NSIDE=16, device=self.device, model_class='conv2d').to(self.device)        
            print('N. total parameters 2D : {0}'.format(sum(p.numel() for p in self.model_2d.parameters() if p.requires_grad)))
            print("=> loading checkpoint '{}'".format(self.checkpoint_2d))
            checkpoint = torch.load(self.checkpoint_2d, map_location=lambda storage, loc: storage)
            self.model_2d.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(self.checkpoint_2d))
            print(f"rho : {checkpoint['state_dict']['rho']}")

    def draw_earth_continents(self):
        coast = []

        for geom in cfeature.COASTLINE.geometries():
            tmp = list(geom.coords)
            n = len(tmp)
            latlon = np.zeros((2,n))
            
            for i in range(n):
                latlon[0, i] = tmp[i][0]
                latlon[1, i] = tmp[i][1]

            coast.append(latlon)

        for i in range(len(coast)):
            hp.visufunc.projplot(coast[i][0,:], coast[i][1,:], 'grey', lonlat=True, linewidth=0.5)


    def evaluate_earth(self, measurement_std=0.001, theta_tik=0.01, cloud_type='earth', year=2004, pov='faceon'):
        
        # Set orbital properties
        p_rotation = 23.934
        p_orbit = 365.256363 * 24.0
        phi_orb = np.pi

        self.pov = pov

        if (self.pov == 'faceon'):
            inclination = 0 * np.pi/180.0 #0.001#np.pi/2
            obliquity = 90. * np.pi/180.0
        
        if (self.pov == 'edgeon'):
            inclination = 90 * np.pi/180.0 #0.001#np.pi/2
            obliquity = 23. * np.pi/180.0

        if (self.pov == 'edgeon_zero'):
            inclination = 90 * np.pi/180.0 #0.001#np.pi/2
            obliquity = 0. * np.pi/180.0

        phi_rot = np.pi/2.0

        nside = 16
        npix = hp.nside2npix(nside)        
        polar_angle, azimuthal_angle = hp.pixelfunc.pix2ang(nside, np.arange(npix))

        x = np.sin(polar_angle) * np.cos(azimuthal_angle)
        y = np.sin(polar_angle) * np.sin(azimuthal_angle)
        z = np.cos(polar_angle)
        
        # From NASA Earth Observations
        tmp = pd.read_csv('earth_720.csv')    
        earth = tmp.values[:, :]

        # earth[earth < 99999.0] = 1.0
        earth[earth == 99999.0] = 0.0
        
        lat = 90. - 0.5*np.arange(359)
        lon = -180 + 0.5*np.arange(720)

        LON, LAT = np.meshgrid(lon, lat)

        ind = hp.ang2pix(nside, LON, LAT, lonlat=True)

        simulated_map = np.zeros(npix)
        simulated_map[ind] = earth        

        simulated_map = hp.sphtfunc.smoothing(simulated_map, fwhm=0.06)
        simulated_map[simulated_map < 0] = 0.0

        # Observation schedule. Each epoch is one week.
        epoch_duration_days = 7
        cadence = 5.0 # hours
        nobs_per_epoch = int(24 // cadence * epoch_duration_days)
        epoch_duration = nobs_per_epoch * cadence
        n_epochs = int(p_orbit / 24.0 / epoch_duration_days)
        epoch_starts = [epoch_duration_days*24*j for j in range(n_epochs)]

        if (cloud_type == 'generate'):
            simulated_clouds = np.zeros((n_epochs, npix))
            n_octaves = 5    

            print("Generating clouds...")
            for k in tqdm(range(n_epochs)):
                seed = 123+k
                noises = [None] * n_octaves
                for i in range(n_octaves):        
                    noises[i] = OpenSimplex(seed=seed+i)
                for i in range(npix):
                    freq = 1.0
                    persistence = 1.0
                    for j in range(n_octaves):
                        simulated_clouds[k, i] += billow_noise(noises[j], x[i], y[i], z[i], freq, persistence)
                        freq *= 1.65
                        persistence *= 0.5

            thr = 0.3
            mx = np.max(simulated_clouds)
            mn = np.min(simulated_clouds)
            simulated_clouds = (simulated_clouds - mn) / (mx - mn)
            simulated_clouds[simulated_clouds < thr] = 0.0

            mx = np.max(simulated_clouds[simulated_clouds > thr])
            mn = np.min(simulated_clouds[simulated_clouds > thr])
            simulated_clouds[simulated_clouds > thr] = (simulated_clouds[simulated_clouds > thr] - mn) / (mx - mn)

        if (cloud_type == 'earth'):
            simulated_clouds = np.zeros((n_epochs, npix))
            lat_clouds = 90. - 180.0 / 64.0 * np.arange(64)
            lon_clouds = -180 + 360.0 / 128.0 * np.arange(128)

            LON_clouds, LAT_clouds = np.meshgrid(lon_clouds, lat_clouds)

            ind_clouds = hp.ang2pix(nside, LON_clouds, LAT_clouds, lonlat=True)

            print("Generating clouds from Earth observations...")
            d = datetime.date(year, 1, 1)
            for k in tqdm(range(n_epochs)):
                clouds = np.loadtxt(f'clouds_earth/T42_{d.year}.{d.month:02d}.{d.day:02d}_davg.dat')
                simulated_clouds[k, ind_clouds] = 0.7 * (clouds / 100.0)**1
                d += datetime.timedelta(days=epoch_duration_days)

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

        largest_eval = scipy.sparse.linalg.eigsh(Phi.T @ Phi, k=1, which='LM', return_eigenvectors=False)       
        rho = 0.4 / largest_eval[0]
                        
        Phi_split = Phi.reshape((n_epochs, nobs_per_epoch, npix))
        
        d_split = np.zeros((n_epochs, nobs_per_epoch))
        for i in range(n_epochs):
            d_split[i, :] = Phi_split[i, : ,:] @ (simulated_clouds[i, :] + (1.0 - simulated_clouds[i, :])**2 * simulated_map) + measurement_std * np.random.randn(nobs_per_epoch)
            # d_split[i, :] = Phi_split[i, : ,:] @ (simulated_map + (0.7 - simulated_map) * simulated_clouds[i, :] / 0.7) + measurement_std * np.random.randn(nobs_per_epoch)

        self.surf0 = torch.zeros((1, 3072)).to(self.device)
        self.clouds0 = torch.zeros((1, n_epochs, 3072)).to(self.device)
        Phi_split = torch.tensor(Phi_split[None, :, :, :].astype('float32')).to(self.device)
        rho = torch.tensor(rho[None].astype('float32')).to(self.device)
        d_split = torch.tensor(d_split[None, :, :].astype('float32')).to(self.device)
                
        self.model_1d.eval()
        self.model_2d.eval()
        
        with torch.no_grad():
                        
            start = time.time()
            surf_1d, clouds_1d, out_surface_1d, out_clouds_1d = self.model_1d(d_split, self.surf0, self.clouds0, Phi_split, rho, n_epochs=n_epochs)            
            print(f'Elapsed time 1D : {time.time()-start}')

            start = time.time()
            surf_2d, clouds_2d, out_surface_2d, out_clouds_2d = self.model_2d(d_split, self.surf0, self.clouds0, Phi_split, rho, n_epochs=n_epochs)            
            print(f'Elapsed time 2D : {time.time()-start}')            
        
        out_surface_1d = out_surface_1d[-1].squeeze().cpu().numpy()
        out_clouds_1d = out_clouds_1d[-1].squeeze().cpu().numpy()
        out_surface_2d = out_surface_2d[-1].squeeze().cpu().numpy()
        out_clouds_2d = out_clouds_2d[-1].squeeze().cpu().numpy()
        Phi_split = Phi_split.cpu().numpy()

        
        # import ipdb
        # ipdb.set_trace()

        return simulated_map, simulated_clouds, out_surface_1d, out_surface_2d, out_clouds_1d, out_clouds_2d, Phi_split

    def doplot_earth_single(self):

        pov = 'faceon'

        f, ax = pl.subplots(nrows=5, ncols=4, figsize=(15,12))

        space = 0.03
        pos_up = ax[0,0].get_position().bounds
        pos_down = ax[-1,0].get_position().bounds
        delta = (pos_up[1] + pos_up[2] - pos_down[1] - space) / 5.0

        for i in range(5):
            for j in range(4):
                pts = ax[i,j].get_position().bounds
                if (i >= 1):
                    new_pts = [pts[0], pos_up[1] - i * delta - space, delta, pts[3]]
                else:
                    new_pts = [pts[0], pos_up[1] - i * delta, delta, pts[3]]
                if (j == 0):
                    print(pts, '->',  new_pts)
                ax[i,j].set_position(new_pts)
            
        simulated_map,simulated_clouds, out_surface_1d, out_surface_2d, out_clouds_1d, out_clouds_2d, Phi_np = self.evaluate_earth(measurement_std=0.001, pov=pov)

                       
        pl.axes(ax[0,0])
        hp.mollview(simulated_map, hold=True, title='Original', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()
        pl.axes(ax[0,1])
        hp.mollview(out_surface_1d, hold=True, title='Reconstructed 1D', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()
        pl.axes(ax[0,2])
        hp.mollview(out_surface_2d, hold=True, title='Reconstructed 2D', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()
        pl.axes(ax[0,3])
        weight = np.sum(Phi_np, axis=(0,1,2))
        hp.mollview(weight / np.max(weight), hold=True, title='Weight', cmap=pl.cm.viridis, rot=0, flip='geo')
        
        indices = np.floor(np.linspace(0,51,8)).astype('int')
        loop = 0
        for i in range(2):
            for j in range(4):
                week = indices[loop]
                pl.axes(ax[1+2*i, j])
                hp.mollview(simulated_clouds[week, :], hold=True, title=f'Original clouds week {week}', cmap=pl.cm.viridis, rot=0, flip='geo')
                pl.axes(ax[1+2*i+1, j])
                hp.mollview(out_clouds_2d[week, :], hold=True, title=f'Inferred clouds week {week}', cmap=pl.cm.viridis, rot=0, flip='geo')
                loop += 1

            
        pl.show()

        pl.savefig(f'earth_clouds_realistic_{pov}.pdf', bbox_inches='tight')

        simulated_clouds_mean = np.mean(simulated_clouds, axis=0)
        simulated_clouds_std = np.std(simulated_clouds, axis=0)
        
        weight = np.mean(Phi_np, axis=2)[0, :, :]
        out_clouds_2d_mean, out_clouds_2d_std = weighted_avg_and_std(out_clouds_2d, weight, axis=0)

        breakpoint()

        f, ax = pl.subplots(nrows=2, ncols=2, figsize=(10,7))
        pl.axes(ax[0,0])
        hp.mollview(simulated_clouds_mean, hold=True, title='Original clouds mean', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()

        pl.axes(ax[0,1])
        hp.mollview(simulated_clouds_std / simulated_clouds_mean, hold=True, title='Original clouds variability', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()

        pl.axes(ax[1,0])
        hp.mollview(out_clouds_2d_mean, hold=True, title='Inferred clouds mean', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()

        pl.axes(ax[1,1])
        hp.mollview(out_clouds_2d_std / out_clouds_2d_mean, hold=True, title='Inferred clouds variability', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()

        pl.show()

        pl.savefig(f'clouds_statistics_{pov}.pdf')

        

            
if (__name__ == '__main__'):
    
    # deepnet = Testing(gpu=2, checkpoint_1d='trained_denoise_clouds/2020-10-15-16:04:03.pth', checkpoint_2d='trained_denoise_clouds/2020-10-15-16:06:54.pth', K1d=15, K2d=15, model_type='denoise')
    deepnet = Testing(gpu=2, checkpoint_1d='trained_denoise_clouds_1d/2020-10-15-16:04:03.pth', checkpoint_2d='trained_denoise_clouds_2d/2020-10-26-16:35:14.pth', K1d=15, K2d=15, model_type='denoise')
    
    deepnet.doplot_earth_single()
    