import os
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.ndimage.filters import gaussian_filter1d

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22

demo_data = []

for i in range(5):
    try:
        demo_file = os.path.join(os.getcwd(), 'final_plots', 'demo_eval{}.log'.format(i+1))

        data = open(demo_file, "r")
        for line in data:
            iter_data = json.loads(line)
            demo_data.append(iter_data['actor_eval'])
    except Exception as e:
        print(e)

sac_data = []

for i in range(5):
    try:
        sac_file = os.path.join(os.getcwd(), 'final_plots', 'sac_eval{}.log'.format(i+1))

        data = open(sac_file, "r")
        for line in data:
            iter_data = json.loads(line)
            sac_data.append(iter_data['actor_eval'])
    except Exception as e:
        print(e)

demo_data = gaussian_filter1d(demo_data, sigma=2)
sac_data = gaussian_filter1d(sac_data, sigma=2)

fig = plt.figure()
ax = fig.gca()  
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')  
handleD, = plt.plot(np.arange(len(demo_data)), demo_data, label='M-DeMoRL', linewidth=2.0)
handleS, = plt.plot(np.arange(len(sac_data)), sac_data, label='SAC',linewidth=2.0)
plt.xlabel('Environment Steps')
plt.ylabel('Episode Performance')
plt.xlim(0,75)
# plt.title(env_name_dict[env])
lgd = plt.legend([handleD, handleS], ['M-DeMoRL', 'SAC'], loc='center left', bbox_to_anchor=(1, 0.5))
lgd.get_frame().set_linewidth(2.0)
plt.savefig(os.path.join(os.getcwd(), 'final_plots', 'base_comparison.png'), bbox_extra_artists=(lgd,), bbox_inches='tight')
