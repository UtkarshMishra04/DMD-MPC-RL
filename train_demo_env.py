import collections
from copy import deepcopy
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
from logger import Logger
from tqdm import tqdm
import pickle
from datetime import datetime

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
        max_step = 1000
    if torch.is_tensor(policy.mean):
        old_mean = policy.mean.clone()
    else:
        old_mean = policy.mean.copy()
    avg_reward = 0.
    avg_cost = 0.
    
    for _ in range(eval_episodes):
        state, done = eval_env.reset(eval=True), False
        i=0
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
    config = construct_config()
    datenow = datetime.now()
    logger_kwargs={'output_dir':'./results/'+ args.exp_name, 'exp_name':datenow.strftime("%m-%d-%H-%M")}
    logger = Logger(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"], use_tb=True)
    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(
        'DeMoRL', args.exp_name, args.seed))
    print("---------------------------------------")

    env = gym.make(args.exp_name)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("State space: {}, Action space: {}".format(state_dim, action_dim))

    replay_buffer = sac.ReplayBuffer(state_dim, action_dim, int(1e6))

    # Choose a controller
    policy, sac_policy, dynamics = get_policy(args,  env, replay_buffer, config)
    sac_policy.ac_targ = deepcopy(sac_policy.ac)

    ckpt = 0

    noise_amount = config['mpc_config']['epsilon']

    total_timesteps = 0
    episode_timesteps = 0
    episode_reward, episode_cost = 0, 0
    evaluation_rewards, evaluation_costs = 0, 0
    evaluation_episodes = 0
    state, done, done_episode = env.reset(), False, False

    if ckpt==0:
        start_instant = 0
    else:
        start_instant = ckpt+1

    for t in tqdm(range(start_instant, int(args.max_timesteps))):
        total_timesteps += 1
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            # action = env.action_space.sample()
            action = policy.get_action(np.array(state))
            action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high)

        else:
            action = policy.get_action(np.array(state))
            action = action + np.random.normal(action.shape) * noise_amount
            action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high)

        # Take the safe action
        next_state, reward, done, info = env.step(action)

        episode_reward += reward
        if 'cost' in info:
            episode_cost += info['cost']

        if hasattr(env, '_max_episode_steps'):
            done_bool = float(
                done) if episode_timesteps < env._max_episode_steps else 0

            if episode_timesteps >= env._max_episode_steps or done:
                done_episode=True
        else:
            done_bool = float(
                done) if episode_timesteps < 1000 else 0

            if episode_timesteps >= 1000 or done:
                done_episode=True

        # Store data in replay buffer
        replay_buffer.store(state, action, reward,next_state,  done_bool, cost=info.get('cost',0))
        state = next_state

        if (t+1) % args.dynamics_freq == 0:
            dynamics_trainloss, dynamics_valloss = dynamics.train()

        if t >= args.start_timesteps and t%sac_policy.update_every==0:
            sac_policy.train()

        if done_episode:
            policy.reset()
            evaluation_costs += episode_cost
            evaluation_rewards += episode_reward
            episode_reward, episode_cost = 0, 0
            evaluation_rewards, evaluation_costs = 0,0
            evaluation_episodes += 1
            state, done = env.reset(), False
            done_episode=False
            episode_timesteps = 0

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0 and t > args.start_timesteps:

            actor_rew = eval_policy_actor(sac_policy, env)
            logger.log('eval/step', t, t)
            logger.log('eval/episode_reward', actor_rew, t)
            logger.dump(t)
            pickle.dump((replay_buffer.state, replay_buffer.action, replay_buffer.reward, replay_buffer.next_state, replay_buffer.cost, replay_buffer.done), open('{}/replay_buffer_{}.pkl'.format(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"], str(t)), 'wb'))
            torch.save(sac_policy.ac.state_dict(), '{}/sac_policy_{}.pt'.format(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"], str(t)))
            torch.save(dynamics.model.ensemble_model.state_dict(), '{}/dynamics_{}.pt'.format(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"], str(t)))
            pickle.dump((dynamics.model.elite_model_idxes, dynamics.model.scaler), open('{}/scalar_trasform_{}.pkl'.format(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"], str(t)), 'wb'))
            evaluation_rewards, evaluation_episodes, evaluation_costs = 0, 0, 0

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
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=1e3, type=int)
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--max_timesteps", default=1e5, type=int)
    parser.add_argument("--dynamics_freq", default=250, type=int)
    parser.add_argument("--exp_name", default="HalfCheetah-v2")
    args = parser.parse_args()
    run_loop(args)