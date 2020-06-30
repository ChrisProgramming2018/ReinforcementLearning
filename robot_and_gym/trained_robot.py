import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import robosuite as suite
from agent_average_v1 import TD31v1



def createstate(state):
    all_states = np.array([])
    for key  in state.keys():
        all_states = np.concatenate((all_states, state[key]))
    return all_states



def main(args):
    """ Starts different tests

    Args:
        param1(args): args

    """
    env = suite.make(
            args.env_name,
            has_renderer=True,           # noon-screenrenderer
            has_offscreen_renderer=True, # no off-screen renderer
            use_object_obs=True,          # usebject-centric feature
            use_camera_obs=False,         # no camera 
            reward_shaping=True,)

    state = env.reset()
    state = createstate(state)
    state_dim = state.shape[0]
    action_dim = env.dof
    max_action = float(6)
    min_action = float(-6)
    print(state_dim)
        
    policy = TD31v1(state_dim, action_dim, max_action, args) 
    directory = 'pytorch_models'
    filename ="TD3_" +  args.env_name + '_42-' + args.agent
    print("Load %s/%s_actor.pth" % (directory, filename))
    policy.load(filename, directory)
    avg_reward = 0.
    seeds = [x for x in range(2)]
    for s in seeds:
        state = env.reset()
        done = False
        while not done:
            state = createstate(state)
            action = policy.select_action(state)
            #action = action.clip(-1, 1)
            #action = np.random.randn(env.dof)
            #print(action)
            state , reward, done, _ = env.step(action)
            avg_reward += reward
            env.render()
    avg_reward /= len(seeds)
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print ("---------------------------------------")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="SawyerLift", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--start_timesteps', default=25e3, type=int)
    parser.add_argument('--eval_freq', default=10000, type=int)  # How often the evaluation step is performed (after how many timesteps)
    parser.add_argument('--repeat_opt', default=1, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--max_timesteps', default=2e6, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr-critic', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr-actor', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--save_model', default=True, type=bool)     # Boolean checker whether or not to save the pre-trained model
    parser.add_argument('--expl_noise', default=0.1, type=float)      # Exploration noise - STD value of exploration Gaussian noise
    parser.add_argument('--batch_size', default= 256, type=int)      # Size of the batch
    parser.add_argument('--discount', default=0.99, type=float)      # Discount factor gamma, used in the calculation of the total discounted reward
    parser.add_argument('--tau', default=0.005, type= float)        # Target network update rate
    parser.add_argument('--policy_noise', default=0.2, type=float)   # STD of Gaussian noise added to the actions for the exploration purposes
    parser.add_argument('--noise_clip', default=0.5, type=float)     # Maximum value of the Gaussian noise added to the actions (policy)
    parser.add_argument('--policy_freq', default=2, type=int)         # Number of iterations to wait before the policy network (Actor model) is updated
    parser.add_argument('--target_update_freq', default=50, type=int)
    parser.add_argument('--num_q_target', default=4, type=int)    # amount of qtarget nets
    parser.add_argument('--train_every_step', default=True, type=bool)    # amount of qtarget nets
    parser.add_argument('--tensorboard_freq', default=5000, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--device', default='cuda', type=str)    # amount of qtarget nets
    parser.add_argument('--run', default=1, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--agent', default="301", type=str)    # load the weights saved after the given number 
    arg = parser.parse_args()
    main(arg)
