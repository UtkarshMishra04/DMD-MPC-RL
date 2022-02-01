
# Dynamic Mirror Descent based Model Predictive Control for Accelerating Robot Learning

### To be presented at the 2022 International Conference on Robotics and Automation (ICRA)

**Authors**: [Utkarsh A. Mishra](https://utkarshmishra04.github.io)\*, [Soumya R. Samineni](https://soumyarani.github.io/)\*, Prakhar Goel, Chandravaran Kunjeti, Himanshu Lodha, Aman Singh, Aditya Sagi, [Shalabh Bhatnagar](https://www.csa.iisc.ac.in/~shalabh/) and [Shishir Kolathaya](https://shishirny.github.io/)

(\* Equal Contribution)

**Paper**: [arxiv.org/pdf/2112.02999.pdf](https://arxiv.org/pdf/2112.02999.pdf) 

**Video**: [https://youtu.be/Bj9dN1KNPAs](https://youtu.be/Bj9dN1KNPAs) 
  

**Abstract**: Recent works in Reinforcement Learning (RL) combine model-free (Mf)-RL algorithms with model-based (Mb)-RL approaches to get the best from both: asymptotic performance of Mf-RL and high sample-efficiency of Mb-RL. Inspired by these works, we propose a hierarchical framework that integrates online learning for the Mb-trajectory optimization with off-policy methods for the Mf-RL. In particular, two loops are proposed, where the Dynamic Mirror Descent based Model Predictive Control (DMD-MPC) is used as the inner loop Mb-RL to obtain an optimal sequence of actions. These actions are in turn used to significantly accelerate the outer loop Mf-RL. We show that our formulation is generic for a broad class of MPC-based policies and objectives, and includes some of the well-known Mb-Mf approaches. We finally introduce a new algorithm: Mirror-Descent Model Predictive RL (M-DeMoRL), which uses Cross-Entropy Method (CEM) with elite fractions for the inner loop. Our experiments show faster convergence of the proposed hierarchical approach on benchmark MuJoCo tasks. We also demonstrate hardware training for trajectory tracking in a 2R leg and hardware transfer for robust walking in a quadruped. We show that the inner-loop Mb-RL significantly decreases the number of training iterations required in the real system, thereby validating the proposed approach.

<!-- [methodology]: './assets/methodology.jpg'
[legresults]: './assets/legresults.jpg'
[hardresults]: './assets/hardresults.gif'
[simresults]: './assets/simresults.gif'
[stochresults]: './assets/stochresults.gif' -->

<p align="center">
  <img width="70%" src="./assets/methodology.jpg">
</p>

<!-- ![alt text][methodology] -->

### Simulation Results

<p align="center">
  <img width="60%" src="./assets/simresults.gif">
</p>

### Hardware Results

<p align="center">
  <img width="70%" src="./assets/legresults.jpg">
</p>

<p align="center">
  <img width="45%" src="./assets/hardresults.gif"> <img width="45%" src="./assets/stochresults.gif">
</p>

## Usage:

Use `Python 3.6` and install `requirements.txt`:
```
pip install -r requirements.txt
```
### For OpenAI environments:

Run:
```
python train_demo_env.py --exp_name {gym env id}
```
### For Other environments:

The environments directory contains the environments from:

- Dextrous Gym: [https://github.com/henrycharlesworth/dexterous-gym](https://github.com/henrycharlesworth/dexterous-gym)

- PDDM: [https://github.com/google-research/pddm](https://github.com/google-research/pddm)

- Stoch: [https://github.com/StochLab/SlopedTerrainLinearPolicy](https://github.com/StochLab/SlopedTerrainLinearPolicy)

Import the environment of your choice. Example of Stoch requires `PyBullet` and runs with:
```
python train_demo_stoch.py
```

## Citation:

```
@article{mishra2021dynamic,
  title={Dynamic Mirror Descent based Model Predictive Control for Accelerating Robot Learning},
  author={Mishra, Utkarsh A, Samineni, Soumya R, Goel, Prakhar, Kunjeti, Himanshu, Lodha, Aman, Singh, Aditya, Sagi, Shalabh, Bhatnagar, and Kolathaya, Shishir},
  journal={arXiv preprint arXiv:2106.15273},
  year={2021}
}
```

## Acknowledgement:

We thank the authors of:

- [LOOP](https://github.com/hari-sikchi/LOOP), [MBPO-PyTorch](https://github.com/Xingyu-Lin/mbpo_pytorch) and [Stable Baselines](https://github.com/DLR-RM/stable-baselines3) for the structured open source code.
- Dextrous Gym, PDDM and Stoch Lab for their open source robot environments
