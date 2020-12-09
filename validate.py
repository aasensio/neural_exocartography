import numpy as np
import matplotlib.pyplot as pl
import torch
import model
import glob
import os
from opensimplex import OpenSimplex
import exocartographer
import healpy as hp


def simplex_noise(noise, x, y, z, freq, weight):
    return weight * (noise.noise3d(x*freq, y*freq, z*freq) / 2.0 + 0.5)


class Testing(object):
    def __init__(self, gpu=0, checkpoint=None, K=3, model_class='conv1d'):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        self.K = K
        
        if (checkpoint is None):
            files = glob.glob('trained/*.pth')
            self.checkpoint = max(files, key=os.path.getctime)
        else:
            self.checkpoint = '{0}'.format(checkpoint)

        if (model_class == 'conv1d'):
            self.model = model.Network(K=self.K, L=32, device=self.device, model_class=model_class).to(self.device)
        
        if (model_class == 'conv2d'):
            self.model = model.Network(K=self.K, L=32, NSIDE=16, device=self.device, model_class=model_class).to(self.device)
            
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        print("=> loading checkpoint '{}'".format(self.checkpoint))

        checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(self.checkpoint))

        print(f"rho : {torch.exp(checkpoint['state_dict']['rho'])}")
        print(f"theta     : {torch.exp(checkpoint['state_dict']['theta'])}")

    def test(self):

        pl.close('all')

        seed = np.random.randint(low=0, high=10000)
        noise = OpenSimplex(seed=137)

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

        Phi = torch.tensor(truth.visibility_illumination_matrix(p)[None, :, :].astype('float32')).to(self.device)
        PhiT = torch.transpose(Phi, 1, 2)

        light = truth.lightcurve(p)

        light = torch.tensor(light[None, :, None].astype('float32')).to(self.device)

        np.random.seed(123)
        self.x0 = torch.tensor(np.random.rand(1, 3072).astype('float32')).to(self.device)
        self.x0 = torch.zeros((1, 3072)).to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():

            out, _ = self.model(light, self.x0, Phi, PhiT)
        
        
        simulated_light = (Phi @ out[:, :, None]).squeeze().cpu().numpy()
        light = light.squeeze().cpu().numpy()
        out = out.squeeze().cpu().numpy()
        
        hp.mollview(simulated_map, cmap=pl.cm.viridis)
        hp.mollview(out, cmap=pl.cm.viridis)

        f, ax = pl.subplots()
        ax.plot(light)
        ax.plot(simulated_light)

        pl.show()

            
if (__name__ == '__main__'):
    
    deepnet = Testing(gpu=0, checkpoint=None, K=9, model_class='conv1d')

    deepnet.test()