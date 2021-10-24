import environments
import numpy as np
import dmc2gym

from envs.environments.dexterous_gym.envs.egg_hand_over import EggHandOver
from envs.environments.dexterous_gym.envs.block_hand_over import BlockHandOver
from envs.environments.dexterous_gym.envs.pen_hand_over import PenHandOver
from envs.environments.dexterous_gym.envs.pen_spin import PenSpin

from envs.environments.mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0
from envs.environments.mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0
from envs.environments.mj_envs.hand_manipulation_suite.pen_v0 import PenEnvV0
from envs.environments.mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

from envs.environments.pddm.envs.dclaw.dclaw_turn_env import DClawTurnEnv
from envs.environments.pddm.envs.cube.cube_env import CubeEnv

from envs.environments.stoch_env.stochlite_pybullet_env import StochliteEnv

'''
Information regarding valid envs:

############################################################
For universe = 'dmc'
Domain:                 Task:               Action Repeat:

cartpole                swingup             8
cheetah                 run                 4
hopper                  hop                 4
finger                  spin                2
finger                  turn                2
ball_in_cup             catch               2
walker                  walk                2
walker                  run                 2
humanoid                walk                2
humanoid                run                 2 

For universe = 'custom'
Domain:                 Task:               Action Repeat:

manip                   door                -
manip                   hammer              -
manip                   pen                 -
manip                   relocate            -

dexter                  egg                 -
dexter                  block               -
dexter                  pen                 -
dexter                  penspin             -

pddm                    claw                -
pddm                    cube                -

stoch                   flat                -
stoch                   upslope             -
stoch                   downslope           -
stoch                   noisy               -

############################################################

'''


def get_stoch_env(task, render=False):

    if task == 'flat':
        env = StochliteEnv(render=render,
                        end_steps = 500, 
                        wedge=False, 
                        stairs = False, 
                        downhill= False, 
                        seed_value=123,
                        on_rack=False, 
                        gait = 'trot',
                        IMU_Noise=False)

    elif task == 'upslope':
        env = StochliteEnv(render=render,
                        end_steps = 500, 
                        wedge=True, 
                        stairs = False, 
                        downhill= False, 
                        seed_value=123,
                        on_rack=False, 
                        gait = 'trot',
                        IMU_Noise=False)
    elif task == 'downslope':
        env = StochliteEnv(render=render,
                        end_steps = 500, 
                        wedge=True, 
                        stairs = False, 
                        downhill= True, 
                        seed_value=123,
                        on_rack=False, 
                        gait = 'trot',
                        IMU_Noise=False)
    elif task == 'noisy':
        env = StochliteEnv(render=render,
                        end_steps = 500, 
                        wedge=False, 
                        stairs = False, 
                        downhill= False, 
                        seed_value=123,
                        on_rack=False, 
                        gait = 'trot',
                        IMU_Noise=True)
    return env

def get_manipulation_env(task):

    if task == 'door':
        env = DoorEnvV0()
    elif task == 'hammer':
        env = HammerEnvV0()
    elif task == 'pen':
        env = PenEnvV0()
    elif task == 'relocate':
        env = RelocateEnvV0()

    return env

def get_dextrous_env(task):

    if task == 'egg':
        env = EggHandOver()
    elif task == 'block':
        env = BlockHandOver()
    elif task == 'pen':
        env = PenHandOver()
    elif task == 'spin':
        env = PenSpin()

    return env

def get_pddm_env(task):

    if task == 'claw':
        env = DClawTurnEnv()
    elif task == 'cube':
        env = CubeEnv()

    return env


def make_env(universe, domain, task, seed=123, action_repeat=2, render=False):

    if universe == 'dmc':
        env = dmc2gym.make(
            domain_name=domain,
            task_name=task,
            resource_files=None,
            img_source=None,
            total_frames=1000,
            seed=seed,
            visualize_reward=False,
            from_pixels=False,
            frame_skip=action_repeat
        )
    else:
        if domain=='manip':
            env = get_manipulation_env(task=task)
        elif domain=='dexter':
            env = get_dextrous_env(task=task)
        elif domain=='pddm':
            env = get_pddm_env(task=task)
        elif domain=='stoch':
            env = get_stoch_env(task=task, render=render)
        else:
            assert False, 'Please enter valid domain-task'

    return env