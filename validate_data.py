from os import stat
import numpy as np
import pickle
import torch
from numpy.core.numeric import tensordot
# from model import EnsembleDynamicsModel
from envs.hardware_env.stoch3_kinematics import Serial2RKinematics
from tqdm import tqdm
from sac import ReplayBuffer

#base_dir = '/home/utkubuntu/Spring_2021/StochLab/DeMoRL/code/hardware/Learning/LOOP/StochP2P-v2'
base_dir = '/home/stoch-lab/Stoch3Leg/Stoch3_icra2021/Stoch3/Learning/LOOP/StochEnv-v2'
log_dir = base_dir + '/09-15-03-55'
model_file = log_dir + '/replay_buffer_21499.pkl'

env_pool = ReplayBuffer(7, 2)
with open(model_file, 'rb') as handle:
    env_pool.state, env_pool.action, env_pool.reward, env_pool.next_state, env_pool.cost, env_pool.done  = pickle.load(handle)
    env_pool.ptr = len(env_pool.state)
    env_pool.size = len(env_pool.state)

# env_model = EnsembleDynamicsModel(7, 5, 4, 2, 1, 200, use_decay=True)

# for i in range(100):
#     state, action, reward, next_state, done = env_pool.sample(2000)
#     delta_state = next_state[:,1:5] - state[:,1:5]
#     inputs = np.concatenate((state[:,1:5], action[:,1:5]), axis=-1)
#     labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)
#     loss = env_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)
#     print("Epoch: {} | Loss: {}".format(i, loss))

# torch.save(env_model.ensemble_model.state_dict(), base_dir + "/env_model.pt")
# pickle.dump((env_model.scaler, env_model.elite_model_idxes), open('{}/scalar_transform.pkl'.format(base_dir), 'wb'))

data_dict = env_pool.sample_batch(None)
state, action, reward, next_state, done = data_dict['obs'].cpu().numpy(), data_dict['act'].cpu().numpy(), data_dict['rew'].cpu().numpy(), data_dict['obs2'].cpu().numpy(), data_dict['done'].cpu().numpy()
# print(state)
# inputs = np.concatenate((state[0], action[0]), axis=-1)
# ensemble_model_means, _ = env_model.predict(inputs)

# print("Error:", np.mean(ensemble_model_means, axis=0)[0][1:] + state[0][:,1:5] - state[1][:,1:5], np.mean(ensemble_model_means, axis=0)[0][0] - reward[0])

kinematics = Serial2RKinematics()

all_phases = []
all_q1pos = []
all_q2pos = []
all_q1vel = []
all_q2vel = []
all_q1tau = []
all_q2tau = []
all_xs = []
all_ys = []
all_xdes = []
all_ydes = []
leg_step_length = 0.16
z_foot = -0.44
step_height = 0.1

for iter in tqdm(range(19999)):
    obs = state[iter]
    all_phases.append(obs[0])
    all_q1pos.append(obs[1])
    all_q2pos.append(obs[2])
    # all_q1vel.append(obs[3])
    # all_q2vel.append(obs[4])
    #all_q1tau.append(obs[5])
    #all_q2tau.append(obs[6])

    [x, y] = kinematics.forwardKinematics([obs[1], obs[2]])
    all_xs.append(x)
    all_ys.append(y)
    # obs[0]=np.clip(obs[0],0,2*np.pi)

    if obs[0] < np.pi:
        xdes = leg_step_length * np.cos(obs[0])
        ydes = z_foot
    else:
        xdes = leg_step_length * np.cos(2*np.pi - obs[0])
        ydes = z_foot + step_height * np.sin(2*np.pi - obs[0])

    all_xdes.append(xdes)
    all_ydes.append(ydes)

print(np.array(done).shape)

all_rewards = []
episode_reward = 0
transition_indices = []
max_reward = -1
for i in range(len(done)):
    episode_reward += reward[i]
    transition_indices.append(i)
    if done[i]:
        all_rewards.append(episode_reward)
        if max_reward < episode_reward:
            max_reward = episode_reward
            all_transition_indices = transition_indices
        transition_indices = []
        episode_reward = 0

# print(all_rewards)
# print(max_reward, all_transition_indices)

start = 2500
dstart = 700
import matplotlib.pyplot as plt
plt.plot(all_phases) #[start:start+dstart])
plt.plot(np.array(done)[start:start+dstart])
plt.ylabel('Traj')
plt.show()
# for dstart in range(200):
#     dstart=dstart*10
#     import matplotlib.pyplot as plt
#     plt.plot(all_xs[start:start+dstart], all_ys[start:start+dstart])
#     plt.plot(all_xdes[start:start+dstart], all_ydes[start:start+dstart])    
#     plt.ylabel('Traj')
#     plt.show()

all_xdes=np.array(all_xdes)
all_xs=np.array(all_xs)
all_xerror=np.abs(all_xdes-all_xs)
all_ydes=np.array(all_ydes)
all_ys=np.array(all_ys)
all_yerror=np.abs(all_ydes-all_ys)

import matplotlib.pyplot as plt
plt.plot(all_xs[start:start+dstart], all_ys[start:start+dstart])
plt.plot(all_xdes[start:start+dstart], all_ydes[start:start+dstart])  
plt.ylabel('Traj')
plt.show()

import matplotlib.pyplot as plt
plt.plot(all_xerror[start:start+dstart])
plt.plot(all_yerror[start:start+dstart])
plt.ylabel('abs error Traj')
plt.show()

   


# print(all_phases[0:10])



plt.plot(np.degrees(all_q1pos[start:start+dstart]))
plt.plot(np.degrees(all_q2pos[start:start+dstart]))
plt.ylabel('Traj')
plt.show()
# print(state[-1])

# plt.plot(all_q1tau[start:start+dstart])
# plt.plot(all_q2tau[start:start+dstart])
# plt.ylabel('Traj')
# plt.show()

# all_xdes = []
# all_ydes = []

# phase = np.linspace(0, 2*np.pi, 100)

# for ph in phase:
#     if ph < np.pi:
#         xdes = leg_step_length * np.cos(ph)
#         ydes = z_foot
#     else:
#         xdes = leg_step_length * np.cos(2*np.pi - ph)
#         ydes = z_foot + step_height * np.sin(2*np.pi - ph)

#     all_xdes.append(xdes)
#     all_ydes.append(ydes)
#     _, q1, q2 = kinematics.inverseKinematics([xdes, ydes])
#     print(ph, xdes, ydes, q1, q2)
#     assert False

# import matplotlib.pyplot as plt
# plt.plot(all_xdes, all_ydes)
# plt.ylabel('Traj')
# plt.show()
ph = np.pi + np.pi/6
if ph < np.pi:
    xdes = leg_step_length * np.cos(ph)
    ydes = z_foot
else:
    xdes = leg_step_length * np.cos(2*np.pi - ph)
    ydes = z_foot + step_height * np.sin(2*np.pi - ph)

all_xdes.append(xdes)
all_ydes.append(ydes)
_, q1, q2 = kinematics.inverseKinematics([xdes, ydes])
print(ph, xdes, ydes, q1, q2)