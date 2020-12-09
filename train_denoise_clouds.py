import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.functional as tf
import torch.utils.data
import time
from tqdm import tqdm
import model_denoise_clouds as model
import argparse
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
import sys
import os
import pathlib
import zarr

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, n_training):
        """
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()

        self.n_training = n_training
        
        f_matrix = zarr.open('training_matrices.zarr', 'r')
        self.matrix = f_matrix['matrix'][:]
        self.eigenvals = f_matrix['largest_eval'][:]
        
        n_samples_matrix, _, _ = self.matrix.shape

        f_surface = zarr.open('training_surfaces_libnoise.zarr', 'r')
        self.surface = 1.0 - f_surface['surface'][:]
        n_samples_surface, _ = self.surface.shape

        f_clouds = zarr.open('training_clouds.zarr', 'r')
        self.clouds = f_clouds['clouds'][:]
        n_samples_clouds, _ = self.clouds.shape

        self.index_matrix = np.random.randint(low=0, high=n_samples_matrix, size=self.n_training)
        self.index_surface = np.random.randint(low=0, high=n_samples_surface, size=self.n_training)
        self.index_clouds = np.random.randint(low=0, high=n_samples_clouds, size=(5, self.n_training))
        
    def __getitem__(self, index):
        
        Phi = self.matrix[self.index_matrix[index], :, :].astype('float32')
        rho = 0.4 / self.eigenvals[self.index_matrix[index]]
        Phi_split = Phi.reshape((5, 24, 3072))
        surface = np.random.uniform(low=0.2, high=1.0) * self.surface[self.index_surface[index], :]
        clouds = np.random.uniform(low=0.2, high=1.0, size=5)[:, None] * self.clouds[self.index_clouds[:, index], :]

        d_split = np.zeros((5, 24))
        for i in range(5):
            d_split[i, :] = Phi_split[i, :, :] @ (clouds[i, :] + (1.0 - clouds[i, :])**2 * surface)            

        return Phi_split, surface.astype('float32'), clouds.astype('float32'), rho.astype('float32'), d_split.astype('float32')

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')
        

class Training(object):
    def __init__(self, batch_size, validation_split=0.2, gpu=0, smooth=0.05, K=3, model_class='conv1d'):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.smooth = smooth
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        # self.device = 'cpu'
        self.batch_size = batch_size
        self.model_class = model_class
        
        self.K = K
        
        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.validation_split = validation_split        
                
        kwargs = {'num_workers': 4, 'pin_memory': False} if self.cuda else {}        
        
        if (model_class == 'conv1d'):
            self.model = model.Network(K=self.K, L=32, device=self.device, model_class=model_class).to(self.device)
        
        if (model_class == 'conv2d'):
            self.model = model.Network(K=self.K, L=32, NSIDE=16, device=self.device, model_class=model_class).to(self.device)
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.train_dataset = Dataset(n_training=20000)
        self.validation_dataset = Dataset(n_training=2000)
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, **kwargs)
        
    def init_optimize(self, epochs, lr, weight_decay, scheduler):

        self.lr = lr
        self.weight_decay = weight_decay        
        print('Learning rate : {0}'.format(lr))
        self.n_epochs = epochs

        if (self.model_class == 'conv1d'):
            root = 'trained_denoise_clouds_1d'
        if (self.model_class == 'conv2d'):
            root = 'trained_denoise_clouds_2d'
        
        p = pathlib.Path(f'{root}/')
        p.mkdir(parents=True, exist_ok=True)

        current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
        self.out_name = f'{root}/{current_time}'

        # Copy model
        file = model.__file__.split('/')[-1]
        shutil.copyfile(model.__file__, '{0}_model.py'.format(self.out_name))
        shutil.copyfile('{0}/{1}'.format(os.path.dirname(os.path.abspath(__file__)), file), '{0}_trainer.py'.format(self.out_name))
        self.file_mode = 'w'

        f = open('{0}_call.dat'.format(self.out_name), 'w')
        f.write('python ' + ' '.join(sys.argv))
        f.close()
        
        f = open('{0}_hyper.dat'.format(self.out_name), 'w')
        f.write('Learning_rate       Weight_decay     \n')
        f.write('{0}    {1}'.format(self.lr, self.weight_decay))
        f.close()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.loss_fn = nn.MSELoss().to(self.device)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler, gamma=0.5)

        np.random.seed(123)
        self.surf0 = torch.tensor(np.random.rand(self.batch_size, 3072).astype('float32')).to(self.device)
        self.surf0 = torch.zeros((self.batch_size, 3072)).to(self.device)

        self.clouds0 = torch.tensor(np.random.rand(self.batch_size, 5, 3072).astype('float32')).to(self.device)
        self.clouds0 = torch.zeros((self.batch_size, 5, 3072)).to(self.device)

        torch.backends.cudnn.benchmark = True
        
    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = 1e10

        trainF = open('{0}.loss.csv'.format(self.out_name), self.file_mode)

        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):            
            self.train(epoch)
            self.test(epoch)
            self.scheduler.step()

            trainF.write('{},{},{}\n'.format(
                epoch, self.loss[-1], self.loss_val[-1]))
            trainF.flush()

            is_best = self.loss_val[-1] < best_loss
            best_loss = min(self.loss_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, filename='{0}.pth'.format(self.out_name))

        trainF.close()
        
    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        n = 1
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (Phi_split, surface, clouds, rho, d_split) in enumerate(t):
            Phi_split, surface, clouds, rho, d_split = Phi_split.to(self.device), surface.to(self.device), clouds.to(self.device), rho.to(self.device), d_split.to(self.device)
            
            self.optimizer.zero_grad()
            
            surf, clouds, out_surface, out_clouds = self.model(d_split, self.surf0, self.clouds0, Phi_split, rho, n_epochs=5)
                                    
            # Loss
            loss = 0.0
            for i in range(self.K):
                loss += self.loss_fn(out_surface[i], surface)
                # loss += self.loss_fn(out_clouds[i], clouds)
                    
            loss.backward()

            self.optimizer.step()

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            if (NVIDIA_SMI):
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                t.set_postfix(loss=loss_avg, lr=current_lr, gpu=tmp.gpu, mem=tmp.memory)
            else:
                t.set_postfix(loss=loss_avg, lr=current_lr)
            
        self.loss.append(loss_avg)

    def test(self, epoch):
        self.model.eval()
        t = tqdm(self.validation_loader)
        n = 1
        loss_avg = 0.0

        with torch.no_grad():
            for batch_idx, (Phi_split, surface, clouds, rho, d_split) in enumerate(t):
                Phi_split, surface, clouds, rho, d_split = Phi_split.to(self.device), surface.to(self.device), clouds.to(self.device), rho.to(self.device), d_split.to(self.device)

                surf, clouds, out_surface, out_clouds = self.model(d_split, self.surf0, self.clouds0, Phi_split, rho, n_epochs=5)
                                                    
                # Loss
                loss = 0.0
                for i in range(self.K):
                    loss += self.loss_fn(out_surface[i], surface)
                    # loss += self.loss_fn(out_clouds[i], clouds)

                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                
                t.set_postfix(loss=loss_avg)
            
        self.loss_val.append(loss_avg)

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--wd', '--weigth-decay', default=0.0, type=float,
                    metavar='WD', help='Weigth decay')    
    parser.add_argument('--gpu', '--gpu', default=0, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=100, type=int,
                    metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--scheduler', '--scheduler', default=100, type=int,
                    metavar='SCHEDULER', help='Number of epochs before applying scheduler')
    parser.add_argument('--batch', '--batch', default=32, type=int,
                    metavar='BATCH', help='Batch size')
    parser.add_argument('--model', '--model', default='conv1d', type=str,
                    metavar='MODEL', help='Model class')
    parser.add_argument('--k', '--k', default=15, type=int,
                    metavar='K', help='K')
    
    parsed = vars(parser.parse_args())

    deepnet = Training(batch_size=parsed['batch'], gpu=parsed['gpu'], smooth=parsed['smooth'], K=parsed['k'], model_class=parsed['model'])

    deepnet.init_optimize(parsed['epochs'], lr=parsed['lr'], weight_decay=parsed['wd'], scheduler=parsed['scheduler'])
    deepnet.optimize()