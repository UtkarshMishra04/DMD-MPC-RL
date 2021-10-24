from sac import ReplayBuffer
import pickle
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from numpy import genfromtxt
from envs.hardware_env.stoch3_kinematics import Serial2RKinematics
from tqdm import tqdm

# my_data = np.loadtxt('/home/stoch-lab/Stoch3Leg/Stoch3_icra2021/Stoch3/Learning/LOOP/PD_Data/data_joints.csv', delimiter=',')
my_data = np.loadtxt('/home/utkubuntu/Spring 2021/StochLab/DeMoRL/Stoch3/Learning/LOOP/PD_Data/data_joints.csv', delimiter=',')

shape = (np.shape(my_data))
# print(shape)
# print_joints_space =1
# dt = 0.008
# if print_joints_space:
#     # plt.plot(np.linspace(0, shape[0]/dt, shape[0]),
#     #          my_data[:, 5], '--r', label='theta_0 tracked')
#     # plt.plot(np.linspace(0, shape[0] / dt, shape[0]),
#     #          my_data[:, 6], 'blue', label='theta_1 tracked')
#     plt.plot(my_data[:, 5],my_data[:, 6])
    
#     leg = plt.legend()
#     plt.show()
# kinematics = Serial2RKinematics()
# all_phases = []
# all_q1pos = []
# all_q2pos = []
# all_q1vel = []
# all_q2vel = []
# all_q1tau = []
# all_q2tau = []
# all_xs = []
# all_ys = []
# all_xdes = []
# all_ydes = []
# all_t0=[]
# all_t1=[]
# leg_step_length = 0.16
# z_foot = -0.44
# step_height = 0.1
# for iter in tqdm(range(160)):
#     obs = my_data[iter]
#     all_phases.append(obs[0])
#     all_q1pos.append(obs[1])
#     all_q2pos.append(obs[2])
#     [x, y] = kinematics.forwardKinematics([obs[1], obs[2]])
#     all_xs.append(x)
#     all_ys.append(y)
#     # obs[0]=np.clip(obs[0],0,2*np.pi)

#     if obs[0] < np.pi:
#         xdes = leg_step_length * np.cos(obs[0])
#         ydes = z_foot
#     else:
#         xdes = leg_step_length * np.cos(2*np.pi - obs[0])
#         ydes = z_foot + step_height * np.sin(2*np.pi - obs[0])
#     [t0_track,t1_track] = kinematics.inverseKinematics([xdes,ydes],branch='<')
#     all_t0.append(t0_track)
#     all_t1.append(t1_track)

#     all_xdes.append(xdes)
#     all_ydes.append(ydes)

# import matplotlib.pyplot as plt
# start=0
# dstart=3000
# plt.plot(all_t0[start:start+dstart])
# plt.plot(all_q1pos[start:start+dstart])  
# # plt.ylabel('Traj')
# plt.show()
# # plt.plot(all_xdes[start:start+dstart],all_ydes[start:start+dstart])
# plt.plot(my_data[:, 5],my_data[:, 6])
# plt.plot(all_xs[start:start+dstart],all_ys[start:start+dstart]) 
# # plt.plot(my_data[:, 9])
# # plt.plot(my_data[:, 10]) 

# plt.ylabel('Traj')
# plt.show()

env_pool = ReplayBuffer(5, 2)
for i in range(my_data.shape[0]-1):
    state = np.array([my_data[i,0],my_data[i,1],my_data[i,2],my_data[i,3],my_data[i,4]],dtype=float)
    action = np.array([my_data[i+1,9],my_data[i+1,10]],dtype=float)
    next_state = np.array([my_data[i+1,0],my_data[i+1,1],my_data[i+1,2],my_data[i+1,3],my_data[i+1,4]],dtype=float)
    reward=100
    terminal=False
    env_pool.store(state,action,reward,next_state,terminal,-reward)
logger_kwargs = {
    "output_dir" : "./PD_Data/" ,
    "exp_name" : "pretrained_v5"
}
t=0
# os.makedirs(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"])
pickle.dump((env_pool.state, env_pool.action, env_pool.reward, env_pool.next_state, env_pool.cost, env_pool.done), open('{}/env_pool_{}.pkl'.format(logger_kwargs["output_dir"]+'/'+logger_kwargs["exp_name"], str(t)), 'wb'))