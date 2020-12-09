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
            module_list = []
            for i in range(self.K):            
                layer1 = nn.Conv1d(1, L, kernel_size=3, padding=1, bias=False)
                nn.init.kaiming_normal_(layer1.weight)
                layer2 = nn.Conv1d(L, L, kernel_size=3, padding=1, bias=False)
                nn.init.kaiming_normal_(layer2.weight)
                layer3 = nn.Conv1d(L, L, kernel_size=3, padding=1, bias=False)
                nn.init.kaiming_normal_(layer3.weight)
                layer4 = nn.Conv1d(L, 1, kernel_size=3, padding=1, bias=False)
                nn.init.kaiming_normal_(layer4.weight)
                module_list.append(nn.ModuleList([layer1, layer2, layer3, layer4]))

        if (model_class == 'conv2d'):
            print("Using convolutions in 2D")
            module_list = []
            for i in range(self.K):            
                layer1 = spherical.sphericalConv(NSIDE, 1, L, bias=False, nest=False)
                layer2 = spherical.sphericalConv(NSIDE, L, L, bias=False, nest=False)
                layer3 = spherical.sphericalConv(NSIDE, L, L, bias=False, nest=False)
                layer4 = spherical.sphericalConv(NSIDE, L, 1, bias=False, nest=False)
                module_list.append(nn.ModuleList([layer1, layer2, layer3, layer4]))

        self.networks = nn.ModuleList(module_list)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        
        self.rho = nn.Parameter(torch.ones(self.K, requires_grad=True))

        self.loss_l2 = nn.MSELoss().to(self.device)


    def prior(self, x, index=None):        
        out = self.relu(self.networks[index][0](x[:, None, :]))
        out = self.relu(self.networks[index][1](out))
        out = self.relu(self.networks[index][2](out))
        out = x[:, None, :] + self.relu(self.networks[index][3](out))
        
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

    def forward(self, y, x0, Phi, PhiT, rho_base):

        nbatch = y.shape[0]
                                
        x = x0.clone()

        d = torch.matmul(PhiT, y)

        out = [None] * self.K

        # x = d.clone().squeeze(-1)
        
        for i in range(self.K):
            t1 = torch.matmul(Phi, x[:, :, None])
            t2 = torch.matmul(PhiT, t1)
            r = x[:, :, None] - rho_base[:, None, None] * self.rho[i].clamp(min=0.1, max=3.0) * (t2 - d)
            x = self.prior(r.squeeze(-1), index=i).view(nbatch, -1)
            out[i] = x
            
        return x, out