# Policy definitions for Online, Offline and Safe RL

from controllers import demorl
import numpy as np
import gym
import core
from models.model_PETS import EnsembleDynamicsModel
from models.predict_env_pets import PredictEnv as PredictEnvPETS
import sac
import torch

# Default termination function that outputs done=False
def default_termination_function(state,action,next_state):
    if (torch.is_tensor(next_state)):
        done = torch.zeros((next_state.shape[0],1))
    else:
        done = np.zeros((next_state.shape[0],1))
    return done

def termination_function(obs, act, next_obs):
    if torch.is_tensor(next_obs):
        done = torch.tensor([False]).repeat(next_obs.shape[0]).view(-1,1)
    else:
        done = np.array([False]).repeat(next_obs.shape[0])
        done = done[:,None]
    return done


def get_policy(args, env, replay_buffer, config):
    policy, sac_policy = None, None
    mpc_config = config['mpc_config']
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    env_model = EnsembleDynamicsModel(7, 5, state_dim, action_dim, 1, 200,
                                        use_decay=True)
    dynamics = PredictEnvPETS(env_model,replay_buffer, args.exp_name, 'pytorch')       

    sac_policy = sac.SAC(env, dynamics, replay_buffer, termination_function)
    sac_policy.update_every=50
    sac_policy.update_after=1000
    policy = demorl.DeMoRL(
        env,
        dynamics,
        sac_policy,termination_function)

    policy.horizon = mpc_config['horizon']
    policy.sol_dim = env.action_space.shape[0] * mpc_config['horizon']
    policy.ub = np.repeat(env.action_space.high, mpc_config['horizon'],axis=0)
    policy.lb = np.repeat(env.action_space.low, mpc_config['horizon'],axis=0)
    policy.mean = np.zeros((policy.sol_dim,))
    policy.N = mpc_config['DeMo']['popsize']
    policy.mixture_coefficient = mpc_config['DeMo']['mixture_coefficient']
    policy.particles = mpc_config['DeMo']['particles']
    policy.max_iters = mpc_config['DeMo']['max_iters']
    policy.alpha = mpc_config['DeMo']['alpha']


    return policy,sac_policy, dynamics
