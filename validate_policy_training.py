from os import stat
import numpy as np
import pickle
import torch
from numpy.core.numeric import tensordot
from models.model_PETS import EnsembleDynamicsModel
from models.predict_env_pets import PredictEnv as PredictEnvPETS
from envs.hardware_env.stoch3_kinematics import Serial2RKinematics
from envs.hardware_env.tracking_env import HardwareEnv
from envs.environments.stoch_env.stochlite_pybullet_env import StochliteEnv
from tqdm import tqdm
from sac import SAC, ReplayBuffer
import os

# base_dir = '/home/utkubuntu/Spring_2021/StochLab/DeMoRL/code/hardware/Learning/LOOP/PD_Data'
base_dir = '/home/stoch-lab/Stoch3Leg/Stoch3_icra2021/Stoch3/Learning/LOOP/PD_Data'
log_dir = base_dir
model_file = log_dir + '/log.csv'

env_pool = ReplayBuffer(3, 2)
data = np.loadtxt(model_file, delimiter=',')[13*710:14*710,:]

for i in range(len(data)-1):
    # print(data[i,:])
    if np.isnan(np.sum(data[i,:])) or np.isnan(np.sum(data[i+1,:])):
        print("Detected Nan")
        pass
    else:
        state = np.array([data[i,14]%2*np.pi, data[i,2], data[i,3]])#, data[i,10], data[i,11]])
        action = np.clip(np.array([data[i,12], data[i,13]])/0.4,-1,1)
        next_state = np.array([data[i+1,14]%2*np.pi, data[i+1,2], data[i+1,3]])#, data[i+1,10], data[i+1,11]])
        reward = np.exp(-10*np.linalg.norm(np.array([data[i,4], data[i,5]]) - np.array([data[i,6], data[i,7]])))

        env_pool.store(state, action, reward, next_state, False)

print(env_pool.size)

env_model = EnsembleDynamicsModel(7, 5, 3, 2, 1, 200, use_decay=True)
dynamics = PredictEnvPETS(env_model, env_pool, 'StochEnv', 'pytorch')       
sac_policy = SAC(StochliteEnv(), dynamics, env_pool, None)
sac_policy.update_every=50
sac_policy.update_after=1000

for i in range(20000):
    # dynamics_trainloss, dynamics_valloss = dynamics.train()
    # sac_policy.train()
    batch = env_pool.sample_batch(1000)
    state, action = batch['obs'], batch['act']
    pred_action, _ = sac_policy.ac.pi(state)
    print(pred_action[0])
    loss = torch.nn.MSELoss()(pred_action, action)
    sac_policy.pi_optimizer.zero_grad()
    loss.backward()
    sac_policy.pi_optimizer.step()
    eval_action,_ =sac_policy.ac.pi(torch.Tensor([[0.501,-1.22067, 1.7012]]))
    print("Iter", i, "Dynamics Train Loss: ", loss.item(), "pred_action", eval_action.cpu().detach().numpy()[0])
    # print("Iter", i, "Dynamics Train Loss: ", dynamics_trainloss, "Dynamics Val Loss: ", dynamics_valloss)

t = 0
logger_kwargs = {'output_dir': base_dir, 'exp_name': 'pretrained_v4'}

if ~os.path.exists(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"]):
    os.makedirs(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"])    

pickle.dump((env_pool.state, env_pool.action, env_pool.reward, env_pool.next_state, env_pool.cost, env_pool.done), open('{}/replay_buffer_{}.pkl'.format(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"], str(t)), 'wb'))
torch.save(sac_policy.ac.state_dict(), '{}/sac_policy_{}.pt'.format(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"], str(t)))
# torch.save(dynamics.model.ensemble_model.state_dict(), '{}/dynamics_{}.pt'.format(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"], str(t)))
# pickle.dump((dynamics.model.elite_model_idxes, dynamics.model.scaler), open('{}/scalar_trasform_{}.pkl'.format(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"], str(t)), 'wb'))
            
