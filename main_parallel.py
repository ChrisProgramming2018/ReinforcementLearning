import numpy as np
import torch
import gym
import argparse
import os
from collections import deque
import utils
import TD3
import OurDDPG
import DDPG
import TD3_ad
import robosuite as suite
from torch.utils.tensorboard import SummaryWriter
import time 
import multiprocessing as mp
from functools import partial



def createstate(state):
    all_states = np.array([])
    for key  in state.keys():
        all_states = np.concatenate((all_states, state[key]))
    return all_states



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=2):
    eval_env = suite.make(
            args.env,
            has_renderer=False,           # noon-screen renderer
            has_offscreen_renderer=False, # no off-screen renderer
            use_object_obs=True,          # use object-centric feature
            use_camera_obs=False,         # no camera 
            reward_shaping=True,
            )
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        for step in range(200):
            state = createstate(state)
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            reward  = reward * 10
            avg_reward += reward 

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def time_format(sec):
    """
    
    Args:
    param1
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, secs

def main(args, param):
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    scores_window = deque(maxlen=100)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = suite.make(
            args.env,
            has_renderer=False,           # noon-screen renderer
            has_offscreen_renderer=False, # no off-screen renderer
            use_object_obs=True,          # use object-centric feature
            use_camera_obs=False,         # no camera 
            reward_shaping=True,
            )
    max_episode_steps = 200
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state = env.reset()
    state = createstate(state)
    state_dim = state.shape[0]

    action_dim = env.dof
    max_action = 1
    pathname = str(args.env) + '_batch_size_' + str(args.batch_size) + '_start_learn_' + str(args.start_timesteps)
    pathname +=  "_seed_" + str(args.seed) + "_state_dim_" + str(state_dim)
    pathname += "num_q_target" + str(param) + "updat_freq" + str(args.target_update_freq)
    print(pathname)
    tensorboard_name = 'runs/' + pathname
    writer = SummaryWriter(tensorboard_name)
    kwargs = {
            "state_dim": state_dim,
        "action_dim": action_dim,
    "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    kwargs["num_q_target"] = param
    policy = TD3_ad.TD3_ad(**kwargs)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]
    state, done = env.reset(), 0
    state = createstate(state)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    tb_update_counter = 0
    t0 = time.time()
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        tb_update_counter += 1
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = np.random.randn(env.dof)
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                    ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        next_state = createstate(next_state)
        done_bool = float(done) if episode_timesteps < max_episode_steps else 0
        if max_episode_steps == episode_timesteps:
            done= 1
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        reward = reward * args.reward_scaling
        episode_reward += reward
        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)
        
        if t % args.target_update_freq == 0:
            policy.hardupdate()
            #pass
        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            duration = time_format(time.time()-t0)
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Time: {duration} ")
            # Reset environment
            if tb_update_counter > args.tensorboard_freq:
                print("tensorboard")
                tb_update_counter = 0
                scores_window.append(episode_reward)
                average_mean = np.mean(scores_window)
                writer.add_scalar('Reward', episode_reward, t)
                writer.add_scalar('Reward mean ', average_mean, t)

            state, done = env.reset(), False
            state = createstate(state)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
        # Evaluate episode
        
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}-" + str(t))
            print("save model")








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="SawyerLift")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=25e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)            # Frequency of delayed policy updates
    parser.add_argument("--tensorboard_freq", default=50, type=int)     # Frequency of delayed policy updates
    parser.add_argument("--target_update_freq", default=500, type=int)   #  Frequency of hard updates
    parser.add_argument("--reward_scaling", default=10, type=int)        # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")             # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                      # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    processes = []
    pool = mp.Pool()
    seeds = [4, 8, 12, 16]
    func = partial(main, args)
    pool.map(func, seeds)
    pool.close()

