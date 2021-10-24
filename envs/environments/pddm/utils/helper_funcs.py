# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import math

import environments.pddm.envs
from envs.environments.pddm.envs.gym_env import GymEnv
from envs.environments.pddm.envs.mb_env import MBEnvWrapper
from envs.environments.pddm.utils.data_structures import *

## calculate angle difference and return radians [-pi, pi]
def angle_difference(x, y):
    angle_difference = np.arctan2(np.sin(x - y), np.cos(x - y))
    return angle_difference