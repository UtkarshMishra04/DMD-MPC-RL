from os import stat
import numpy as np
import pickle
import torch
from numpy.core.numeric import tensordot
from models.model_PETS import EnsembleDynamicsModel
from envs.hardware_env.stoch3_kinematics import Serial2RKinematics
from tqdm import tqdm
from sac import ReplayBuffer

base_dir = '/home/utkubuntu/Spring_2021/StochLab/DeMoRL/code/hardware/Learning/LOOP/StochSAC-v2'
# base_dir = '/home/stoch-lab/Stoch3Leg/Stoch3_icra2021/Stoch3/Learning/LOOP/StochSAC-v2'
log_dir = base_dir + '/09-14-00-54'
model_file = log_dir + '/replay_buffer_42999.pkl'

env_pool = ReplayBuffer(7, 2)
with open(model_file, 'rb') as handle:
    env_pool.state, env_pool.action, env_pool.reward, env_pool.next_state, env_pool.cost, env_pool.done  = pickle.load(handle)
    env_pool.ptr = len(env_pool.state)
    env_pool.size = len(env_pool.state)

env_model = EnsembleDynamicsModel(7, 5, 3, 2, 1, 200, use_decay=True)

# batch = env_pool.sample_batch(None)
# state, action, reward, next_state, done = batch['obs'], batch['act'], batch['rew'], batch['obs2'], batch['done']

for i in range(100):
    batch = env_pool.sample_batch(2000)
    state, action, reward, next_state, done = batch['obs'], batch['act'], batch['rew'], batch['obs2'], batch['done']
    delta_state = next_state - state
    inputs = np.concatenate((state[:,:3], action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state[:,:3]), axis=-1)
    loss = env_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)
    print("Epoch: {} | Loss: {}".format(i, loss))

# torch.save(env_model.ensemble_model.state_dict(), base_dir + "/env_model.pt")
env_model.ensemble_model.load_state_dict(torch.load(base_dir + "/env_model_3.pt", map_location='cpu'))
# pickle.dump((env_model.scaler, env_model.elite_model_idxes), open('{}/scalar_transform.pkl'.format(base_dir), 'wb'))
env_model.scaler, env_model.elite_model_idxes = pickle.load(open('{}/scalar_transform_3.pkl'.format(base_dir), 'rb'))

# test_index = np.linspace(0, 42000, 100).astype(int)
# input = np.concatenate((state[test_index, :3], action[test_index]), axis=-1)
# outputs, _ = env_model.predict(input)
# # output = np.mean(outputs, axis=0)

# for i in range(7):
#     output = outputs[i]
#     print(state[test_index, :3].numpy().shape, output[:,1:].shape, next_state[test_index, :3].numpy().shape)
#     reward_error = np.mean(np.abs(reward[test_index].numpy() - output[:,0]), axis=0)
#     state_error = np.mean(np.abs(state[test_index, :3].numpy() + output[:,1:] - next_state[test_index, :3].numpy()), axis=0)
#     print("Reward error: {}".format(reward_error), "State error: {}".format(state_error))