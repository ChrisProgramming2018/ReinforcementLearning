import torch
import torchx as tx
import torchx.nn as nnx
import torch.nn as nn
import torchx.layers as L
import numpy as np

class CNNStemNetwork(nnx.Module):
    def __init__(self, D_obs, D_out, conv_channels=[16, 32], kernel_sizes=[8, 4], strides=[4,2]):
        super(CNNStemNetwork, self).__init__()
        layers = []
        for i in range(len(conv_channels)):
            layers.append(L.Conv2d(conv_channels[i], kernel_size=kernel_sizes[i], stride=strides[i]))
            layers.append(L.ReLU())
        layers.append(L.Flatten())
        layers.append(L.Linear(D_out))
        layers.append(L.ReLU())
        self.model = L.Sequential(*layers)

        # instantiate parameters
        self.model.build((None, *D_obs))

    def forward(self, obs):
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 5)
        if if_high_dim: # case of RNN input
            obs = obs.view(-1, *obs_shape[2:])  

        obs = self.model(obs)

        if if_high_dim:
            obs = obs.view(obs_shape[0], obs_shape[1], -1)
        return obs
