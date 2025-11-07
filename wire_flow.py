#!/usr/bin/env python

import os
import sys
from tqdm import tqdm
import importlib
import time

import numpy as np
from scipy import io

import matplotlib.pyplot as plt
plt.gray()

import cv2
from skimage.metrics import structural_similarity as ssim_func

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR
from pytorch_msssim import ssim

from modules import models
from modules import utils

if __name__ == '__main__':
    nonlin = 'wire'            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = 2000               # Number of SGD iterations
    learning_rate = 5e-3        # Learning rate. 
    
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3 
    
    tau = 3e1                   # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2               # Readout noise (dB)
    
    # Gabor filter constants.
    # We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
    omega0 = 5.0           # Frequency of sinusoid
    sigma0 = 5.0           # Sigma of Gaussian
    
    # Network parameters
    hidden_layers = 2       # Number of hidden layers in the MLP
    hidden_features = 256   # Number of hidden units per layer
    maxpoints = 256*256     # Batch size
    
    # Read image and scale. A scale of 0.5 for parrot image ensures that it
    # fits in a 12GB GPU
    import scipy.io as sio
    flow_dict = sio.loadmat('data/flow_mat.mat')
    H, W,  = 256, 256

    
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
        
        if tau < 100:
            sidelength = int(max(H, W)/3)
        else:
            sidelength = int(max(H, W))
            
    else:
        posencode = False
        sidelength = H

        
    model = models.get_INR(
                    nonlin=nonlin,
                    in_features=2,
                    out_features=3, 
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    first_omega_0=omega0,
                    hidden_omega_0=omega0,
                    scale=sigma0,
                    pos_encode=posencode,
                    sidelength=sidelength)
        
    # Send model to CUDA
    model.cuda()
    
    coords_sparse = torch.from_numpy(flow_dict['coords_sparse']).float()
    print(coords_sparse.shape)
    coords_dense = torch.from_numpy(flow_dict['coords_dense'])
    sparse_flow_rgb = torch.from_numpy(flow_dict['sparse_flow_rgb']).float().cuda() / 1.
    flow_rgb = torch.from_numpy(flow_dict['flow_rgb']).float()
    vasc_bin = flow_dict['vasc_bin']
    print('Number of parameters: ', utils.count_parameters(model))

    # Create an optimizer
    optim = torch.optim.Adam(lr=learning_rate*min(1, maxpoints/(coords_sparse.shape[1])),
                             params=model.parameters())
    
    # Schedule to reduce lr to 0.1 times the initial rate in final epoch
    scheduler = LambdaLR(optim, lambda x: 0.1**min(x/niters, 1))
    
    # x = torch.linspace(-1, 1, W)
    # y = torch.linspace(-1, 1, H)
    
    # X, Y = torch.meshgrid(x, y, indexing='xy')
    # coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
    


    
    
    
    mse_array = torch.zeros(niters, device='cuda')
    mse_loss_array = torch.zeros(niters, device='cuda')
    time_array = torch.zeros_like(mse_array)
    
    best_mse = torch.tensor(float('inf'))
    best_img = None
    
    
    
    tbar = tqdm(range(niters))
    init_time = time.time()
    for epoch in tbar:
        indices = torch.randperm(coords_sparse.shape[1])
        
        for b_idx in range(0, coords_sparse.shape[1], maxpoints):
            b_indices = indices[b_idx:min(coords_sparse.shape[1], b_idx+maxpoints)]
            b_coords = coords_sparse[:, b_indices, ...].cuda()
            b_indices = b_indices.cuda()
            pixelvalues = model(b_coords)
            
            # with torch.no_grad():
            #     rec[:, b_indices, :] = pixelvalues
    
            loss = ((pixelvalues - sparse_flow_rgb[:, b_indices, :])**2).mean() 
            
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        time_array[epoch] = time.time() - init_time
        
        with torch.no_grad():
            flow_pred = 255.*np.ones((256,256,3))
            indices = torch.linspace(0,coords_dense.shape[1]-1,coords_dense.shape[1]).int()
            preds = []
            for b_idx in range(0, coords_dense.shape[1], maxpoints):
                b_indices = indices[b_idx:min(coords_dense.shape[1], b_idx+maxpoints)]
                b_coords = coords_dense[:, b_indices, ...].cuda()
                b_indices = b_indices.cuda()
                pixelvalues = model(b_coords)
                preds.append(pixelvalues)
            predicted_flow = torch.cat(preds,1)
            flow_pred[vasc_bin > 0] = predicted_flow[0,::].detach().cpu().numpy()

        
        
            tbar.refresh()
        
        scheduler.step()
        
        
            
        cv2.imshow('Reconstruction', (1.*flow_pred).astype(np.uint8)[..., ::-1])            
        cv2.waitKey(1)
    
        if (mse_array[epoch] < best_mse) or (epoch == 0):
            best_mse = mse_array[epoch]
            best_img = flow_pred
    
    if posencode:
        nonlin = 'posenc'
        
    mdict = {'rec': best_img,
             'mse_noisy_array': mse_loss_array.detach().cpu().numpy(), 
             'mse_array': mse_array.detach().cpu().numpy(),
             'time_array': time_array.detach().cpu().numpy()}
    
    os.makedirs('results/denoising', exist_ok=True)
    io.savemat('results/denoising/%s.mat'%nonlin, mdict)
    import skimage
    skimage.io.imsave('best_flow.png',(1.*flow_pred).astype(np.uint8))
