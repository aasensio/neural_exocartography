import torch
import torch.nn as nn
import numpy as np
import spherical

def prox_soft(X, step):
    """Soft thresholding proximal operator    
    """    
    return torch.sign(X) * nn.functional.relu(torch.abs(X) - step)

class Network(nn.Module):
    def __init__(self, L=32, K=9, NSIDE=16, device='cpu', model_class='conv1d'):
        super(Network, self).__init__()
        
        self.K = K
        self.L = L
        self.device = device

        if (model_class == 'conv1d'):
            print("Using convolutions in 1D")
            module_list_surface = []
            for i in range(self.K):            
                layer1 = nn.Conv1d(1, L, kernel_size=3, padding=1, bias=False)
                nn.init.kaiming_normal_(layer1.weight)
                layer2 = nn.Conv1d(L, L, kernel_size=3, padding=1, bias=False)
                nn.init.kaiming_normal_(layer2.weight)
                layer3 = nn.Conv1d(L, L, kernel_size=3, padding=1, bias=False)
                nn.init.kaiming_normal_(layer3.weight)
                layer4 = nn.Conv1d(L, 1, kernel_size=3, padding=1, bias=False)
                nn.init.kaiming_normal_(layer4.weight)
                module_list_surface.append(nn.ModuleList([layer1, layer2, layer3, layer4]))


        if (model_class == 'conv2d'):
            print("Using convolutions in 2D")
            module_list_surface = []
            for i in range(self.K):            
                layer1 = spherical.sphericalConv(NSIDE, 1, L, bias=False, nest=False)
                layer2 = spherical.sphericalConv(NSIDE, L, L, bias=False, nest=False)
                layer3 = spherical.sphericalConv(NSIDE, L, L, bias=False, nest=False)
                layer4 = spherical.sphericalConv(NSIDE, L, 1, bias=False, nest=False)
                module_list_surface.append(nn.ModuleList([layer1, layer2, layer3, layer4]))

        self.networks_surface = nn.ModuleList(module_list_surface)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        
        self.rho = nn.Parameter(torch.ones(self.K, requires_grad=True))

        self.loss_l2 = nn.MSELoss().to(self.device)


    def prior_surface(self, x, index=None):        
        out = self.relu(self.networks_surface[index][0](x[:, None, :]))
        out = self.relu(self.networks_surface[index][1](out))
        out = self.relu(self.networks_surface[index][2](out))
        out = x[:, None, :] + self.relu(self.networks_surface[index][3](out))
        
        return out.clamp(min=0.0, max=1.0)

    def prior_clouds(self, x, index=None):        
        out = self.relu(self.networks_clouds[index][0](x[:, None, :]))
        out = self.relu(self.networks_clouds[index][1](out))
        out = self.relu(self.networks_clouds[index][2](out))
        out = x[:, None, :] + self.relu(self.networks_clouds[index][3](out))
        
        return out.clamp(min=0.0, max=1.0)

    def loss_forward_back(self, x):

        loss = 0.0
        for index in range(self.K):
            out = self.act(self.networks[index][0](x[:, None, :]))
            out = self.networks[index][1](out)
            
            out = self.act(self.networks[index][2](out))
            out = self.networks[index][3](out)

            loss += self.loss_l2(out.squeeze(), x)
        
        return loss

    def forward(self, d_split, surf0, clouds0, Phi_split, rho_base, n_epochs):

        nbatch = d_split.shape[0]
        npix = surf0.shape[1]
                                
        surf = surf0.clone()
        clouds = clouds0.clone()
    
        out_surface = [None] * self.K
        out_clouds = [None] * self.K
        
        for i in range(self.K):
            grad_surf = surf0.clone()
            grad_clouds = clouds0.clone()
            for j in range(n_epochs):
                t0 = 1.0 - clouds[:, j, :]
                t1 = t0 * t0
                tmp = clouds[:, j, :] + t1 * surf
                Phi = Phi_split[:, j, :, :]
                PhiT = torch.transpose(Phi, 1, 2)                
                tmp = torch.matmul(Phi, tmp[:, :, None])
                t2 = torch.matmul(PhiT, tmp - d_split[:, j, :, None]).squeeze()
                
                grad_surf = grad_surf + 2.0 * t2 * t1
                grad_clouds[:, j, :] = 2.0 * t2 - 4.0 * t2 * surf * t0
            
            r_surf = surf - rho_base[:,  None] * self.rho[i].clamp(min=0.1, max=3.0) * grad_surf
            r_clouds = clouds - rho_base[:, None, None] * self.rho[i].clamp(min=0.1, max=3.0) * grad_clouds

            # Apply the prior projection on the surface
            surf = self.prior_surface(r_surf.squeeze(-1), index=i).view(nbatch, -1)
            
            # Do not use any prior project for the clouds
            clouds = self.relu(r_clouds)
            # clouds = self.prior_surface(r_clouds.squeeze(-1).view(-1, npix), index=i).view(nbatch, n_epochs, -1)
            

            # r_clouds = r_clouds.view(-1, npix)
            # clouds = self.prior_clouds(r_clouds.squeeze(-1), index=i).view(nbatch, 5, npix)

            out_surface[i] = surf
            out_clouds[i] = clouds
            
        return surf, clouds, out_surface, out_clouds