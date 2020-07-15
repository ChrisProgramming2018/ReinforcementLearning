import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchx as tx
import torchx.nn as nnx
import torchx.layers as L


class Actor(nn.Module):

  def __init__(self, state_dim, action_dim, max_action=1):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x))
    return x




class Critic(nn.Module):

  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1

class ActorNetworkX(nnx.Module):
    def __init__(self, D_in, D_act, hidden_sizes=[300, 200], use_layernorm=True):
        super(ActorNetworkX, self).__init__()

        xp_input = L.Placeholder((None, D_in))
        xp = L.Linear(hidden_sizes[0])(xp_input)
        xp = L.ReLU()(xp)
        if use_layernorm:
            # Normalize 1 dimension
            xp = L.LayerNorm(1)(xp)
        xp = L.Linear(hidden_sizes[1])(xp)
        xp = L.ReLU()(xp)
        if use_layernorm:
            xp = L.LayerNorm(1)(xp)
        xp = L.Linear(D_act)(xp)
        xp = L.Tanh()(xp)

        self.model = L.Functional(inputs=xp_input, outputs=xp)
        self.model.build((None, D_in))

    def forward(self, obs):
        return self.model(obs)

class CriticNetworkX(nnx.Module):
    def __init__(self, D_in, D_act, hidden_sizes=[400, 300], use_layernorm=True):
        super(CriticNetworkX, self).__init__()

        xp_input_obs = L.Placeholder((None, D_in))
        xp = L.Linear(hidden_sizes[0])(xp_input_obs)
        xp = L.ReLU()(xp)
        if use_layernorm:
            xp = L.LayerNorm(1)(xp)
        self.model_obs = L.Functional(inputs=xp_input_obs, outputs=xp)
        self.model_obs.build((None, D_in))

        xp_input_concat = L.Placeholder((None, hidden_sizes[0] + D_act))
        xp = L.Linear(hidden_sizes[1])(xp_input_concat)
        xp = L.ReLU()(xp)
        if use_layernorm:
            xp = L.LayerNorm(1)(xp)
        xp = L.Linear(1)(xp)

        self.model_concat = L.Functional(inputs=xp_input_concat, outputs=xp)
        self.model_concat.build((None, D_act + hidden_sizes[0]))

    def forward(self, obs, act):
        h_obs = self.model_obs(obs)
        h1 = torch.cat((h_obs, act), 1)
        value = self.model_concat(h1)
        return value



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

class CNNNetwork(nn.Module):
    def __init__(self, D_obs, D_out, conv_channels=[16, 32], kernel_sizes=[8, 4], strides=[4,2]):
        super(CNNNetwork, self).__init__()
        
        #self.conv = torch.nn.Sequential()
        channels = 3
        #self.conv.add_module("conv_1", torch.nn.Conv2d(channels, conv_channels[0], kernel_sizes[0], strides[0]))
        #self.conv.add_module("relu_1", torch.nn.ReLU())
        #self.conv.add_module("conv_2", torch.nn.Conv2d(conv_channels[0], conv_channels[1], kernel_sizes[1], strides[1]))
        #self.conv.add_module("relu_2", torch.nn.ReLU())
        #self.conv.add_module("flatten", torch.nn.Flatten())
        #self.conv.add_module("Linear", torch.nn.Linear(D_out))
        #self.conv.add_module("relu_3", torch.nn.ReLU())
        
        self.conv_1 =  torch.nn.Conv2d(channels, conv_channels[0], kernel_sizes[0], strides[0])
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 =  torch.nn.Conv2d(conv_channels[0], conv_channels[1], kernel_sizes[1], strides[1])
        self.relu_2 = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.Linear  = torch.nn.Linear(2592, D_out)
        self.relu_3 = torch.nn.ReLU()




    def forward(self, obs):
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 5)
        if if_high_dim: # case of RNN input
            obs = obs.view(-1, *obs_shape[2:])  

        x = self.conv_1(obs)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.flatten(x)
        obs = self.relu_3(self.Linear(x))

        if if_high_dim:
            obs = obs.view(obs_shape[0], obs_shape[1], -1)
        return obs

