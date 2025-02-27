import torch
import torch.nn as nn
import torch.nn.functional as F
import healpy as hp
import matplotlib.pyplot as pl
import numpy as np
import glob

class sphericalConv(nn.Module):
    def __init__(self, NSIDE, in_channels, out_channels, bias=True, nest=True):
        """
        Convolutional neural networks on the HEALPix sphere: a
        pixel-based algorithm and its application to CMB data analysis
        N. Krachmalnicoff1, 2? and M. Tomasi3, 4

        """
        super(sphericalConv, self).__init__()

        self.NSIDE = NSIDE
        self.npix = hp.nside2npix(self.NSIDE)
        
        files = glob.glob('neighbours_*.npy')
        found = False
        for f in files:
            nside = int(f.split('_')[1].split('.')[0])
            if nside == NSIDE:
                found = True                
                self.neighbours = np.load(f)

        if not found:
            print(f"Computing neighbours for NSIDE={NSIDE}")
            self.neighbours = torch.zeros(9 * self.npix, dtype=torch.long)
            for i in range(self.npix):
                # neighbours = [i]
                # neighbours.extend(hp.pixelfunc.get_all_neighbours(self.NSIDE, i, nest=nest))
                
                neighbours = hp.pixelfunc.get_all_neighbours(self.NSIDE, i, nest=nest)
                neighbours = np.insert(neighbours, 4, i)

                neighbours[neighbours == -1] = neighbours[4]

                self.neighbours[9*i:9*i+9] = torch.tensor(neighbours)

            np.save(f'neighbours_{NSIDE}.npy', self.neighbours)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=9, bias=bias)

        nn.init.kaiming_normal_(self.conv.weight)        
        if (bias):
            nn.init.constant_(self.conv.bias, 0.0)
        
    def forward(self, x):
        
        vec = x[:, :, self.neighbours]        
        
        tmp = self.conv(vec)

        return tmp

class sphericalDown(nn.Module):
    def __init__(self, NSIDE):
        super(sphericalDown, self).__init__()
        
        self.pool = nn.AvgPool1d(4)
                
    def forward(self, x):
                
        return self.pool(x)

class sphericalUp(nn.Module):
    def __init__(self, NSIDE):
        super(sphericalUp, self).__init__()
                
    def forward(self, x):
        
        return torch.repeat_interleave(x, 4, dim=-1)

if (__name__ == '__main__'):
    
    NSIDE = 8
    pl.close('all')

    conv = sphericalConv(NSIDE, 1, 1, bias=False, nest=False)
    conv2 = sphericalConv(NSIDE, 1, 1, bias=False, nest=False)
    conv3 = sphericalConv(NSIDE, 1, 1, bias=False, nest=False)
    # down = sphericalDown(NSIDE)
    # up = sphericalUp(NSIDE // 2)

    npix = hp.nside2npix(NSIDE)

    im = torch.zeros(1,1,npix)

    im[0, 0, :] = torch.linspace(0., 1.0, npix)
    # im[0, 0, :] = torch.randn(npix)

    out = conv(im)
    # out = conv2(out)
    # out = conv3(out)
    # out2 = down(out)
    # out3 = up(out2)

    hp.mollview(im[0, 0, :].numpy(), nest=False)
    hp.mollview(out[0, 0, :].detach().numpy(), nest=False)
    # hp.mollview(out3[0, 0, :].detach().numpy(), nest=True)

    # hp.mollview(im[0, 1, :].numpy())

    # hp.mollview(out[0, 0, :].detach().numpy(), nest=True)
    # hp.mollview(out[0, 1, :].detach().numpy())
    pl.show()