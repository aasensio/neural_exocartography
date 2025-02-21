import numpy as np
import matplotlib.pyplot as pl
import torch
import model_denoise as model
import exocartographer
import healpy as hp
import pandas as pd
import time
import scipy.sparse.linalg
from tqdm import tqdm
import datetime
import cartopy.feature as cfeature
from opensimplex import OpenSimplex

def simplex_noise(noise, x, y, z, freq, weight):
    return weight * (noise.noise3(x*freq, y*freq, z*freq) / 2.0 + 0.5)

def billow_noise(noise, x, y, z, freq, weight):
    return weight * (2.0 * np.abs(noise.noise3(x*freq, y*freq, z*freq)) - 1.0)


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


    def evaluate_earth(self, measurement_std=0.001, theta_tik=0.01, cloud_type='none', year=2004, pov='faceon'):
        
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
            out_surface_1d, _ = self.model_1d(light, self.x0, Phi, PhiT, rho)
            print(f'Elapsed time 1D : {time.time()-start}')

            start = time.time()
            out_surface_2d, _ = self.model_2d(light, self.x0, Phi, PhiT, rho)
            print(f'Elapsed time 2D : {time.time()-start}')            
        
        out_surface_1d = out_surface_1d[-1].squeeze().cpu().numpy()        
        out_surface_2d = out_surface_2d[-1].squeeze().cpu().numpy()        
        
        
        # import ipdb
        # ipdb.set_trace()

        return simulated_map, out_surface_1d, out_surface_2d

    def doplot_earth_single(self):

        pov = 'faceon'

        f, ax = pl.subplots(nrows=1, ncols=3, figsize=(15,8))

        space = 0.03
        pos_up = ax[0].get_position().bounds
        pos_down = ax[-1].get_position().bounds
        delta = (pos_up[1] + pos_up[2] - pos_down[1] - space) / 5.0

            
        simulated_map, out_surface_1d, out_surface_2d = self.evaluate_earth(measurement_std=0.001, pov=pov)

                       
        pl.axes(ax[0])
        hp.mollview(simulated_map, hold=True, title='Original', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()
        pl.axes(ax[1])
        hp.mollview(out_surface_1d, hold=True, title='Reconstructed 1D', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()
        pl.axes(ax[2])
        hp.mollview(out_surface_2d, hold=True, title='Reconstructed 2D', cmap=pl.cm.viridis, rot=0, flip='geo')
        self.draw_earth_continents()
                            
        pl.show()

        # pl.savefig(f'earth_clouds_realistic_{pov}.pdf', bbox_inches='tight')
      

            
if (__name__ == '__main__'):
    
    # deepnet = Testing(gpu=2, checkpoint_1d='trained_denoise_clouds/2020-10-15-16:04:03.pth', checkpoint_2d='trained_denoise_clouds/2020-10-15-16:06:54.pth', K1d=15, K2d=15, model_type='denoise')
    # deepnet = Testing(gpu=0, checkpoint_1d='trained_denoise_clouds_1d/2020-10-15-16:04:03.pth', checkpoint_2d='trained_denoise_clouds_2d/2020-10-26-16:35:14.pth', K1d=15, K2d=15, model_type='denoise')

    deepnet = Testing(gpu=0, checkpoint_1d='trained_denoise_1d/2020-10-14-08:23:11.pth', checkpoint_2d='trained_denoise_2d/2020-10-26-16:34:23.pth', K1d=15, K2d=15, model_type='denoise')

    deepnet.doplot_earth_single()
    