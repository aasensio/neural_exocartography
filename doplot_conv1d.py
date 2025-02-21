import numpy as np
import matplotlib.pyplot as pl
import torch
import model_denoise
import exocartographer
import healpy as hp
import tikhonov
import pandas as pd
import time
import scipy.sparse.linalg
import zarr
from PIL import Image
import subprocess
import cartopy.feature as cfeature
from opensimplex import OpenSimplex


def simplex_noise(noise, x, y, z, freq, weight):
    return weight * (noise.noise3d(x*freq, y*freq, z*freq) / 2.0 + 0.5)


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
        
        if (self.model_type == 'sparse'):
            self.model_1d = model.Network(K=self.K1d, L=32, device=self.device, model_class='conv1d').to(self.device)        
            print('N. total parameters 1D : {0}'.format(sum(p.numel() for p in self.model_1d.parameters() if p.requires_grad)))
            print("=> loading checkpoint '{}'".format(self.checkpoint_1d))
            checkpoint = torch.load(self.checkpoint_1d, map_location=lambda storage, loc: storage)
            self.model_1d.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(self.checkpoint_1d))    
            print(f"rho : {torch.exp(checkpoint['state_dict']['rho'])}")
            print(f"theta     : {torch.exp(checkpoint['state_dict']['theta'])}")

            self.model_2d = model.Network(K=self.K2d, L=32, NSIDE=16, device=self.device, model_class='conv2d').to(self.device)        
            print('N. total parameters 2D : {0}'.format(sum(p.numel() for p in self.model_2d.parameters() if p.requires_grad)))
            print("=> loading checkpoint '{}'".format(self.checkpoint_2d))
            checkpoint = torch.load(self.checkpoint_2d, map_location=lambda storage, loc: storage)
            self.model_2d.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(self.checkpoint_2d))
            print(f"rho : {torch.exp(checkpoint['state_dict']['rho'])}")
            print(f"theta     : {torch.exp(checkpoint['state_dict']['theta'])}")

        if (self.model_type == 'denoise'):
            self.model_1d = model_denoise.Network(K=self.K1d, L=32, device=self.device, model_class='conv1d').to(self.device)        
            print('N. total parameters 1D : {0}'.format(sum(p.numel() for p in self.model_1d.parameters() if p.requires_grad)))
            print("=> loading checkpoint '{}'".format(self.checkpoint_1d))
            checkpoint = torch.load(self.checkpoint_1d, map_location=lambda storage, loc: storage)
            self.model_1d.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(self.checkpoint_1d))    
            print(f"rho : {checkpoint['state_dict']['rho']}")

            self.model_2d = model_denoise.Network(K=self.K2d, L=32, NSIDE=16, device=self.device, model_class='conv2d').to(self.device)        
            print('N. total parameters 2D : {0}'.format(sum(p.numel() for p in self.model_2d.parameters() if p.requires_grad)))
            print("=> loading checkpoint '{}'".format(self.checkpoint_2d))
            checkpoint = torch.load(self.checkpoint_2d, map_location=lambda storage, loc: storage)
            self.model_2d.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(self.checkpoint_2d))
            print(f"rho : {checkpoint['state_dict']['rho']}")

    def evaluate(self, seed=137, measurement_std=0.001, theta_tik=0.01):
        
        noise = OpenSimplex(seed=seed)

        # Set orbital properties
        p_rotation = 23.934
        p_orbit = 365.256363 * 24.0
        phi_orb = np.pi
        inclination = 0 * np.pi/180.0 #0.001#np.pi/2
        obliquity = 90. * np.pi/180.0
        phi_rot = np.pi/2.0

        nside = 16
        npix = hp.nside2npix(nside)
        polar_angle, azimuthal_angle = hp.pixelfunc.pix2ang(nside, np.arange(npix))

        x = np.sin(polar_angle) * np.cos(azimuthal_angle)
        y = np.sin(polar_angle) * np.sin(azimuthal_angle)
        z = np.cos(polar_angle)

        simulated_map = np.zeros(npix)
        for i in range(npix):
            simulated_map[i] = simplex_noise(noise, x[i], y[i], z[i], 1.0, 1.0) + \
                simplex_noise(noise, x[i], y[i], z[i], 2.0, 1/4.0) + \
                simplex_noise(noise, x[i], y[i], z[i], 4.0, 1/8.0)
        
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

        # epoch_starts = [15*i*p_rotation for i in range(15)]
        # epoch_starts.extend([19*i*p_rotation for i in range(15)])
        epoch_starts = [30*p_rotation, 60*p_rotation, 150*p_rotation,
                        210*p_rotation, 250*p_rotation]

        times = np.array([])
        for epoch_start in epoch_starts:
            epoch_times = np.linspace(epoch_start,
                                    epoch_start + epoch_duration,
                                    nobs_per_epoch)
            times = np.concatenate([times, epoch_times])

        

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

        Phi_np = truth.visibility_illumination_matrix(p)[None, :, :]

        PhiT_Phi = Phi_np[0, :, :].T @ Phi_np[0, :, :]
        largest_eval = scipy.sparse.linalg.eigsh(PhiT_Phi, k=1, which='LM', return_eigenvectors=False)
        rho = 0.4 / largest_eval
        rho = torch.tensor(rho.astype('float32')).to(self.device)

        Phi = torch.tensor(Phi_np.astype('float32')).to(self.device)
        PhiT = torch.transpose(Phi, 1, 2)
        
        n = len(times)
        light = truth.lightcurve(p)
        light += measurement_std * np.random.randn(n)

        light = torch.tensor(light[None, :, None].astype('float32')).to(self.device)

        np.random.seed(123)
        self.x0 = torch.tensor(np.random.rand(1, 3072).astype('float32')).to(self.device)
        self.x0 = torch.zeros((1, 3072)).to(self.device)
        
        self.model_1d.eval()
        self.model_2d.eval()
        
        with torch.no_grad():

            start = time.time()
            out_1d, _ = self.model_1d(light, self.x0, Phi, PhiT, rho)
            print(f'Elapsed time 1D : {time.time()-start}')
            
            start = time.time()
            out_2d, _ = self.model_2d(light, self.x0, Phi, PhiT, rho)
            print(f'Elapsed time 2D : {time.time()-start}')
        
        
        Phi_np = Phi[0, :, :].cpu().numpy()
        y_np = light[0, :, :].cpu().numpy()
        tik = tikhonov.tikhonov(Phi_np, y_np, theta=theta_tik, iter=1500)
        
        simulated_light_1d = (Phi @ out_1d[:, :, None]).squeeze().cpu().numpy()
        simulated_light_2d = (Phi @ out_2d[:, :, None]).squeeze().cpu().numpy()
        light = light.squeeze().cpu().numpy()
        out_1d = out_1d.squeeze().cpu().numpy()
        out_2d = out_2d.squeeze().cpu().numpy()        

        return simulated_map, out_1d, out_2d, simulated_light_1d, simulated_light_2d, light, tik.flatten()

    def evaluate_libnoise(self, seed=137, measurement_std=0.001, theta_tik=0.01):
        
        noise = OpenSimplex(seed=seed)

        # Set orbital properties
        p_rotation = 23.934
        p_orbit = 365.256363 * 24.0
        phi_orb = np.pi
        inclination = 0 * np.pi/180.0 #0.001#np.pi/2
        obliquity = 90. * np.pi/180.0
        phi_rot = np.pi/2.0

        nside = 16
        npix = hp.nside2npix(nside)

        delta_lat = 180.0 / 512.0
        lat = 90.0 - delta_lat * np.arange(512)

        delta_lon = 360.0 / 1024.0
        lon = delta_lon * np.arange(1024)

        LAT, LON = np.meshgrid(lat, lon)
        
        ind = hp.ang2pix(nside, LON, LAT, lonlat=True)
                
        process = subprocess.run(["./noise", "simple", f"{seed}"])
        
        simulated_map = np.zeros(npix)
        planet = np.array(Image.open(f'planet-{seed}-specular.bmp'))[:, :, 0]

        simulated_map[ind.T] = 1.0 - planet / 255.0

        # simulated_map *= np.random.uniform(low=0.2, high=0.8, size=1)

        process = subprocess.run(["rm", "-f", f"planet-{seed}-specular.bmp"])
        process = subprocess.run(["rm", "-f", f"planet-{seed}-surface.bmp"])
        process = subprocess.run(["rm", "-f", f"planet-{seed}-normal.bmp"])
                        
        # Observation schedule
        cadence = p_rotation/24.
        nobs_per_epoch = 24
        epoch_duration = nobs_per_epoch * cadence

        # epoch_starts = [15*i*p_rotation for i in range(15)]
        # epoch_starts.extend([19*i*p_rotation for i in range(15)])
        epoch_starts = [30*p_rotation, 60*p_rotation, 150*p_rotation,
                        210*p_rotation, 250*p_rotation]

        times = np.array([])
        for epoch_start in epoch_starts:
            epoch_times = np.linspace(epoch_start,
                                    epoch_start + epoch_duration,
                                    nobs_per_epoch)
            times = np.concatenate([times, epoch_times])

        

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

        Phi_np = truth.visibility_illumination_matrix(p)[None, :, :]

        PhiT_Phi = Phi_np[0, :, :].T @ Phi_np[0, :, :]
        largest_eval = scipy.sparse.linalg.eigsh(PhiT_Phi, k=1, which='LM', return_eigenvectors=False)
        rho = 0.4 / largest_eval
        rho = torch.tensor(rho.astype('float32')).to(self.device)

        Phi = torch.tensor(Phi_np.astype('float32')).to(self.device)
        PhiT = torch.transpose(Phi, 1, 2)
        
        n = len(times)
        light = truth.lightcurve(p)
        light += measurement_std * np.random.randn(n)

        light = torch.tensor(light[None, :, None].astype('float32')).to(self.device)

        np.random.seed(123)
        self.x0 = torch.tensor(np.random.rand(1, 3072).astype('float32')).to(self.device)
        self.x0 = torch.zeros((1, 3072)).to(self.device)
        
        self.model_1d.eval()
        self.model_2d.eval()
        
        with torch.no_grad():

            start = time.time()
            out_1d, _ = self.model_1d(light, self.x0, Phi, PhiT, rho)
            print(f'Elapsed time 1D : {time.time()-start}')
            
            start = time.time()
            out_2d, _ = self.model_2d(light, self.x0, Phi, PhiT, rho)
            print(f'Elapsed time 2D : {time.time()-start}')
        
        
        Phi_np = Phi[0, :, :].cpu().numpy()
        y_np = light[0, :, :].cpu().numpy()
        tik = tikhonov.tikhonov(Phi_np, y_np, theta=theta_tik, iter=1500)
        
        simulated_light_1d = (Phi @ out_1d[:, :, None]).squeeze().cpu().numpy()
        simulated_light_2d = (Phi @ out_2d[:, :, None]).squeeze().cpu().numpy()
        light = light.squeeze().cpu().numpy()
        out_1d = out_1d.squeeze().cpu().numpy()
        out_2d = out_2d.squeeze().cpu().numpy()        

        return simulated_map, out_1d, out_2d, simulated_light_1d, simulated_light_2d, light, tik.flatten()
        

    def evaluate_fine(self, seed=137, measurement_std=0.001):
        
        noise = OpenSimplex(seed=seed)

        # Set orbital properties
        p_rotation = 23.934
        p_orbit = 365.256363 * 24.0
        phi_orb = np.pi
        inclination = 0 * np.pi/180.0 #0.001#np.pi/2
        obliquity = 90. * np.pi/180.0
        phi_rot = np.pi/2.0

        nside = 16
        npix = hp.nside2npix(nside)
        polar_angle, azimuthal_angle = hp.pixelfunc.pix2ang(nside, np.arange(npix))

        x = np.sin(polar_angle) * np.cos(azimuthal_angle)
        y = np.sin(polar_angle) * np.sin(azimuthal_angle)
        z = np.cos(polar_angle)

        simulated_map = np.zeros(npix)
        for i in range(npix):
            simulated_map[i] = simplex_noise(noise, x[i], y[i], z[i], 1.0, 1.0) + \
                simplex_noise(noise, x[i], y[i], z[i], 2.0, 1/4.0) + \
                simplex_noise(noise, x[i], y[i], z[i], 4.0, 1/8.0)
        
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

        epoch_starts = [15*i*p_rotation for i in range(25)]
        epoch_starts.extend([19*i*p_rotation for i in range(25)])
        epoch_starts.extend([23*i*p_rotation for i in range(25)])
        # epoch_starts = [30*p_rotation, 60*p_rotation, 150*p_rotation,
                        # 210*p_rotation, 250*p_rotation]

        times = np.array([])
        for epoch_start in epoch_starts:
            epoch_times = np.linspace(epoch_start,
                                    epoch_start + epoch_duration,
                                    nobs_per_epoch)
            times = np.concatenate([times, epoch_times])

        

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

        Phi = torch.tensor(truth.visibility_illumination_matrix(p)[None, :, :].astype('float32')).to(self.device)
        PhiT = torch.transpose(Phi, 1, 2)

        n = len(times)
        light = truth.lightcurve(p)
        light += measurement_std * np.random.randn(n)

        light = torch.tensor(light[None, :, None].astype('float32')).to(self.device)

        np.random.seed(123)
        self.x0 = torch.tensor(np.random.rand(1, 3072).astype('float32')).to(self.device)
        self.x0 = torch.zeros((1, 3072)).to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():

            out, _ = self.model(light, self.x0, Phi, PhiT)
        
        
        Phi_np = Phi[0, :, :].cpu().numpy()
        y_np = light[0, :, :].cpu().numpy()
        tik = tikhonov.tikhonov(Phi_np, y_np, rho=10.0, theta=0.05, iter=1000)
        
        simulated_light = (Phi @ out[:, :, None]).squeeze().cpu().numpy()
        light = light.squeeze().cpu().numpy()
        out = out.squeeze().cpu().numpy()

        

        return simulated_map, out, simulated_light, light, tik.flatten()

    def evaluate_earth(self, measurement_std=0.001, theta_tik=0.01, do_tikhonov=True):
        
        # Set orbital properties
        p_rotation = 23.934
        p_orbit = 365.256363 * 24.0
        phi_orb = np.pi
        inclination = 0 * np.pi/180.0 #0.001#np.pi/2
        obliquity = 90. * np.pi/180.0
        phi_rot = np.pi/2.0

        nside = 16
        npix = hp.nside2npix(nside)
        
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

        # Observation schedule  
        cadence = 5.0  # hours
        n = p_orbit // cadence
        times = cadence * np.arange(n)
        self.times = times
        
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

        Phi_np = truth.visibility_illumination_matrix(p)[None, :, :]

        PhiT_Phi = Phi_np[0, :, :].T @ Phi_np[0, :, :]
        largest_eval = scipy.sparse.linalg.eigsh(PhiT_Phi, k=1, which='LM', return_eigenvectors=False)
        rho = 0.4 / largest_eval
        rho = torch.tensor(rho.astype('float32')).to(self.device)

        Phi = torch.tensor(Phi_np.astype('float32')).to(self.device)
        PhiT = torch.transpose(Phi, 1, 2)

        n = len(times)
        light = truth.lightcurve(p)
        light += measurement_std * np.random.randn(n)

        light = torch.tensor(light[None, :, None].astype('float32')).to(self.device)

        np.random.seed(123)
        self.x0 = torch.tensor(np.random.rand(1, 3072).astype('float32')).to(self.device)
        self.x0 = torch.zeros((1, 3072)).to(self.device)

        self.model_1d.eval()
        self.model_2d.eval()
        
        with torch.no_grad():

            start = time.time()
            out_1d, _ = self.model_1d(light, self.x0, Phi, PhiT, rho)
            print(f'Elapsed time 1D : {time.time()-start}')
            
            start = time.time()
            out_2d, _ = self.model_2d(light, self.x0, Phi, PhiT, rho)
            print(f'Elapsed time 2D : {time.time()-start}')        
        
        Phi_np = Phi[0, :, :].cpu().numpy()
        y_np = light[0, :, :].cpu().numpy()
                
        simulated_light_1d = (Phi @ out_1d[:, :, None]).squeeze().cpu().numpy()
        simulated_light_2d = (Phi @ out_2d[:, :, None]).squeeze().cpu().numpy()
        light = light.squeeze().cpu().numpy()
        out_1d = out_1d.squeeze().cpu().numpy()
        out_2d = out_2d.squeeze().cpu().numpy()

        if (do_tikhonov):
            tik = tikhonov.tikhonov(Phi_np, y_np, theta=theta_tik, iter=2000)
            tik_light = Phi_np @ tik.flatten()
            return simulated_map, out_1d, out_2d, simulated_light_1d, simulated_light_2d, light, tik.flatten(), tik_light
        else:
            return simulated_map, out_1d, out_2d, simulated_light_1d, simulated_light_2d, light, Phi_np

        

        # import ipdb
        # ipdb.set_trace()

        return simulated_map, out_1d, out_2d, simulated_light_1d, simulated_light_2d, light, tik.flatten(), tik_light

    def doplot(self, which=0):

        if (which == 0):
            f, ax = pl.subplots(nrows=4, ncols=4, figsize=(12.5,8.75))        

            seeds = [137, 169, 2560, 3900]

            for i in range(4):            
                simulated_map, out_1d, out_2d, simulated_light_1d, simulated_light_2d, light, tik = self.evaluate_libnoise(seed=seeds[i], measurement_std=0.01, theta_tik=0.5)
                mn = np.mean(light)
                pl.axes(ax[i,0])
                if (i == 0):
                    hp.mollview(simulated_map, hold=True, title='Original', cmap=pl.cm.viridis)
                    pl.axes(ax[i,1])
                    hp.mollview(out_1d, hold=True, title='Reconstructed 1D', cmap=pl.cm.viridis)
                    pl.axes(ax[i,2])
                    hp.mollview(out_2d, hold=True, title='Reconstructed 2D', cmap=pl.cm.viridis)
                    pl.axes(ax[i,3])
                    hp.mollview(tik, hold=True, title='Tikhonov', cmap=pl.cm.viridis)
                else:
                    hp.mollview(simulated_map, hold=True, title='', cmap=pl.cm.viridis)
                    pl.axes(ax[i,1])
                    hp.mollview(out_1d, hold=True, title='', cmap=pl.cm.viridis)
                    pl.axes(ax[i,2])
                    hp.mollview(out_2d, hold=True, title='', cmap=pl.cm.viridis)
                    pl.axes(ax[i,3])
                    hp.mollview(tik, hold=True, title='', cmap=pl.cm.viridis)                                

            pl.tight_layout()
            pl.show()

            pl.savefig(f'several_maps.pdf', bbox_inches='tight')

        if (which == 1):
            f, ax = pl.subplots(nrows=5, ncols=4, figsize=(15,12))

            noises = [0.001, 0.003, 0.01, 0.03, 0.1]
            thetas = [0.05, 0.2, 0.5, 1.0, 2.5]

            for i in range(5):
                simulated_map, out_1d, out_2d, simulated_light_1d, simulated_light_2d, light, tik = self.evaluate_libnoise(seed=137, measurement_std=noises[i], theta_tik=thetas[i])
                mn = np.mean(light)
                
                if (i == 0):
                    pl.axes(ax[i,0])
                    hp.mollview(simulated_map, hold=True, title='Original', cmap=pl.cm.viridis)
                    pl.text(-2,1.0,f'$S/N$={mn/noises[i]:5.1f}')
                    pl.axes(ax[i,1])                    
                    hp.mollview(out_1d, hold=True, title='Reconstructed 1D', cmap=pl.cm.viridis)
                    pl.axes(ax[i,2])
                    hp.mollview(out_1d, hold=True, title='Reconstructed 2D', cmap=pl.cm.viridis)
                    pl.axes(ax[i,3])
                    hp.mollview(tik, hold=True, title='Tikhonov', cmap=pl.cm.viridis)
                else:
                    pl.axes(ax[i,0])
                    hp.mollview(simulated_map, hold=True, title='', cmap=pl.cm.viridis)
                    pl.text(-2,1.0,f'S/N={mn/noises[i]:5.1f}')
                    pl.axes(ax[i,1])
                    hp.mollview(out_1d, hold=True, title='', cmap=pl.cm.viridis)
                    pl.axes(ax[i,2])
                    hp.mollview(out_2d, hold=True, title='', cmap=pl.cm.viridis)
                    pl.axes(ax[i,3])
                    hp.mollview(tik, hold=True, title='', cmap=pl.cm.viridis)                

            pl.tight_layout()
            pl.show()

            pl.savefig(f'noise_influence.pdf', bbox_inches='tight')


    def doplot_examples(self):
        f_surf = zarr.open('/scratch/Dropbox/THEORY/planet_cartography/training_surfaces_libnoise.zarr', 'r')
        f_clouds = zarr.open('/scratch/Dropbox/THEORY/planet_cartography/training_clouds.zarr', 'r')
        f, ax = pl.subplots(nrows=3, ncols=3, figsize=(10,7))

        for i in range(3):
            Ac = f_clouds['clouds'][i, :]
            A = np.random.uniform(low=0.2, high=1.0) * (1.0-f_surf['surface'][i, :])
            pl.axes(ax[0,i])
            hp.mollview(A, hold=True, title='Surface', rot=180, flip='geo', cmap=pl.cm.viridis)
            pl.axes(ax[1,i])
            hp.mollview(Ac, hold=True, title='Clouds', rot=180, flip='geo', cmap=pl.cm.viridis)
            final = Ac + (1.0 - Ac)**2 * A
            pl.axes(ax[2,i])
            hp.mollview(final, hold=True, title='Surface & clouds', rot=180, flip='geo', cmap=pl.cm.viridis)

        pl.tight_layout()
        pl.show()

        pl.savefig('planet_samples.pdf', bbox_inches='tight')

    def doplot_earth(self):
        # f, ax = pl.subplots(nrows=4, ncols=4, figsize=(17,11))
        f, ax = pl.subplots(nrows=4, ncols=4, figsize=(13,9.1))

        noises = [0.0001, 0.001, 0.01, 0.1]
        thetas = [0.005, 0.005, 0.05, 0.1]

        for i in range(len(noises)):
        
            simulated_map, out_1d, out_2d, simulated_light_1d, simulated_light_2d, light, tik, tik_light = self.evaluate_earth(measurement_std=noises[i], theta_tik=thetas[i])
            mn = np.mean(light)
            pl.axes(ax[i,0])
            if (i == 0):
                hp.mollview(simulated_map, hold=True, title='Original', cmap=pl.cm.viridis, rot=0, flip='geo')
            else:
                hp.mollview(simulated_map, hold=True, title='', cmap=pl.cm.viridis, rot=0, flip='geo')
            self.draw_earth_continents()

            pl.text(-2.7,1.0,fr'$\sigma$={noises[i]}')
            pl.text(-2.7,0.8,f'S/N={mn/noises[i]:5.1f}')
            pl.axes(ax[i,1])
            if (i == 0):
                hp.mollview(out_1d, hold=True, title='Reconstructed 1D', cmap=pl.cm.viridis, rot=0, flip='geo')
            else:
                hp.mollview(out_1d, hold=True, title='', cmap=pl.cm.viridis, rot=0, flip='geo')
            self.draw_earth_continents()

            pl.axes(ax[i,2])
            if (i == 0):
                hp.mollview(out_2d, hold=True, title='Reconstructed 2D', cmap=pl.cm.viridis, rot=0, flip='geo')
            else:
                hp.mollview(out_2d, hold=True, title='', cmap=pl.cm.viridis, rot=0, flip='geo')
            self.draw_earth_continents()

            pl.axes(ax[i,3])
            if (i == 0):
                hp.mollview(tik, hold=True, title='Tikhonov', cmap=pl.cm.viridis, rot=0, flip='geo')
            else:
                hp.mollview(tik, hold=True, title='', cmap=pl.cm.viridis, rot=0, flip='geo')
            self.draw_earth_continents()

        pl.tight_layout()
        pl.show()

        pl.savefig('earth.pdf', bbox_inches='tight')


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

    def doplot_earth_single(self):
        
        f, ax = pl.subplots(nrows=1, ncols=4, figsize=(15,4))
    
        simulated_map, out_1d, out_2d, simulated_light_1d, simulated_light_2d, light, Phi_np = self.evaluate_earth(measurement_std=0.001, do_tikhonov=False)
                        
        pl.axes(ax[0])
        hp.mollview(simulated_map, hold=True, title='Original', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()
        pl.axes(ax[1])
        hp.mollview(out_1d, hold=True, title='Reconstructed 1D', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()
        pl.axes(ax[2])
        hp.mollview(out_2d, hold=True, title='Reconstructed 2D', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()
        pl.axes(ax[3])
        weight = np.sum(Phi_np, axis=0)
        hp.mollview(weight / np.max(weight), hold=True, title='Weight', cmap=pl.cm.viridis, rot=0, flip='geo')
        
        pl.tight_layout()
        pl.show()

        pl.savefig('earth.pdf', bbox_inches='tight')

            
if (__name__ == '__main__'):
    
    deepnet = Testing(gpu=0, checkpoint_1d='trained_denoise_1d/2020-10-14-08:23:11.pth', checkpoint_2d='trained_denoise_2d/2020-10-26-16:34:23.pth', K1d=15, K2d=15, model_type='denoise')
    
    deepnet.doplot(which=0)
    # deepnet.doplot(which=1)
    
    # deepnet.doplot_earth()    
    # deepnet.doplot_examples()