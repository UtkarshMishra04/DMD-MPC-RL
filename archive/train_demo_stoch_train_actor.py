# Implements LOOP: ARC with H-step lookahead policies for Online RL

import collections
from operator import le
import numpy as np
import math
import torch
import gym
import argparse
import os
import time
import sac
import yaml
from policies import get_policy
from envs.hardware_env.tracking_env_pd import HardwareEnv
from envs.environments.stoch_env.stochlite_pybullet_env import StochliteEnv
from logger import Logger
from tqdm import tqdm
import pickle
from datetime import datetime

from train_demo_stoch import construct_config

def eval_policy_actor(policy, eval_env, eval_episodes=1):

    avg_reward = 0.
    if hasattr(eval_env, '_max_episode_steps'):
        max_step = eval_env._max_episode_steps
    else:
        max_step = 300

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        episode_steps=0
        while not done:
            episode_steps+=1
            action = policy.get_action(np.array(state),deterministic=True)
            state, reward, done, _ = eval_env.step(action)
            if(episode_steps>=max_step):
                done=True
            avg_reward += reward

    print("---------------------------------------")
    print(
        "Actor| Evaluation over {} episodes: {}".format(
            eval_episodes,
            avg_reward))
    print("---------------------------------------")
    return avg_reward

def eval_policy(policy, eval_env, eval_episodes=1,logger=None):
    
    if hasattr(eval_env, '_max_episode_steps'):
        max_step = eval_env._max_episode_steps
    else:
        max_step = 300
    if torch.is_tensor(policy.mean):
        old_mean = policy.mean.clone()
    else:
        old_mean = policy.mean.copy()
    avg_reward = 0.
    avg_cost = 0.
    
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        i = 0
        policy.reset()
        while not done:
            i+=1
            if(i>=max_step):
                break
            
            action = policy.get_action(np.array(state),deterministic=True)
            next_state, reward, done, info = eval_env.step(action)
            state = next_state
            avg_reward += reward

            if 'cost' in info:
                avg_cost += info['cost']

    policy.mean = old_mean
    print("---------------------------------------")
    print("Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward, avg_cost


def run_loop(args):
    config = construct_config(args)

    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(
        'DeMoRL', args.exp_name, args.seed))
    print("---------------------------------------")

    env = StochliteEnv() #HardwareEnv()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = sac.ReplayBuffer(state_dim, action_dim,int(1e6))

    with open('./PD_Data/pretrained_v5/env_pool_0.pkl', 'rb') as handle:
        replay_buffer.state, replay_buffer.action, replay_buffer.reward, replay_buffer.next_state, replay_buffer.cost, replay_buffer.done  = pickle.load(handle)
        ckpt = 1e6
        for i in range(len(replay_buffer.state)):
            if np.sum(replay_buffer.reward[i:i+100])==0:
                ckpt = i
                break
        replay_buffer.ptr = ckpt
        replay_buffer.size = ckpt
        print("ckpt",ckpt)
        print(replay_buffer.state.shape)
        # assert False

    # Choose a controller
    _, sac_policy, dynamics = get_policy(args,  env, replay_buffer, config)

    for t in tqdm(range(int(args.max_timesteps))):
        sac_policy.train()

    torch.save(sac_policy.ac.state_dict(), './PD_Data/pretrained_v5/sac_policy.pt')
    torch.save(dynamics.model.ensemble_model.state_dict(), './PD_Data/pretrained_v5/dynamics.pt')
    pickle.dump((dynamics.model.elite_model_idxes, dynamics.model.scaler), open('./PD_Data/pretrained_v5/scalar_trasform.pkl', 'wb'))

def construct_config():

    config = {}

    # Environment
    config['mpc_config'] = {}
    mpc_config = {}
    mpc_config['horizon'] = 3
    mpc_config['gamma'] = 0.99
    mpc_config['epsilon'] = 0.0
    
    cem_config = {}
    cem_config['popsize'] = 100
    cem_config['particles'] = 4
    cem_config['actor_mix'] = 5
    cem_config['max_iters'] = 5
    cem_config['num_elites'] = 10
    cem_config['alpha'] = 0.1
    cem_config['mixture_coefficient'] = 0.05

    demo_config = {}
    demo_config['popsize'] = 100
    demo_config['particles'] = 4
    demo_config['max_iters'] = 5
    demo_config['alpha'] = 0.1
    demo_config['mixture_coefficient'] = 0.05

    mpc_config['CEM'] = cem_config
    mpc_config['DeMo'] = demo_config
    config['mpc_config'] = mpc_config

    return config

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="StochEnv-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=1e3, type=int)
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--max_timesteps", default=1e5, type=int)
    parser.add_argument("--dynamics_freq", default=250, type=int)
    parser.add_argument("--exp_name", default="StochEnv-v2")
    args = parser.parse_args()
    run_loop(args)