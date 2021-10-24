import matplotlib.pyplot as plt
import csv
import torch
from envs.hardware_env.tracking_env import HardwareEnv
from envs.environments.stoch_env.stochlite_pybullet_env import StochliteEnvimport numpy as np
from os import stat
import numpy as np
import pickle
from numpy.core.numeric import tensordot
from models.model_PETS import EnsembleDynamicsModel
from models.predict_env_pets import PredictEnv as PredictEnvPETS
from envs.hardware_env.stoch3_kinematics import Serial2RKinematics
from tqdm import tqdm
from sac import SAC, ReplayBuffer
import os

env = StochliteEnv() #HardwareEnv()

env_pool = ReplayBuffer(5, 2)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
env_model = EnsembleDynamicsModel(7, 5, 5, 2, 1, 200, use_decay=True)
dynamics = PredictEnvPETS(env_model, env_pool, 'StochEnv', 'pytorch')       
sac_policy = SAC(StochliteEnv() #HardwareEnv(), dynamics, env_pool, None)
# sac_policy.ac.load_state_dict(torch.load('./PD_Data/pretrained_v4/sac_policy_0.pt', map_location=torch.device('cuda')))
#sac_policy.ac.load_state_dict(torch.load('/home/stoch-lab/Stoch3Leg/Stoch3_icra2021/Stoch3/Learning/LOOP/StochSAC-v2/09-14-20-31/sac_policy_6999.pt', map_location=torch.device('cuda')))
sac_policy.update_every=50
sac_policy.update_after=1000
total_reward = 0
obs=env.reset()

while(1):

    # action,_ = sac_policy.ac.pi(torch.Tensor(obs).unsqueeze(0),deterministic = True)#get_action(obs,deterministic=False)
    # print(action)
    # next_obs, reward, terminal, _ = env.step(action.cpu().detach().numpy()[0])

    # total_reward += reward
    # obs = next_obs

    #print("iter",i,"obs",obs,"reward",reward)
    # action=np.array([1*np.sin(6*time.time()),-1*np.sin(3*time.time())])
    action=[0.3,0.3]
    next_obs, reward, terminal, _ = env.step(action)

    total_reward += reward
    obs = next_obs

    #print(obs)

    if terminal:
        obs=env.reset()
        print("reward=",total_reward)
        total_reward=0



