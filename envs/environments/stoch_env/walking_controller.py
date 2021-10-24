# ### Walking controller
# Written by Shishir Kolathaya shishirk@iisc.ac.in
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for realizing walking controllers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import dataclass
from collections import namedtuple
from custom_envs.stoch_env.utils.ik_class import StochliteKinematics
import numpy as np

PI = np.pi
no_of_points = 100

@dataclass
class leg_data:
    name: str
    ID: int
    motor_hip: float = 0.0
    motor_knee: float = 0.0
    motor_abduction: float = 0.0
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    b: float = 1.0
    step_length: float = 0.0
    x_shift = 0.0
    y_shift = 0.0
    z_shift = 0.0


@dataclass
class robot_data:
    front_left: leg_data = leg_data('FL', 1)
    front_right: leg_data = leg_data('FR', 2)
    back_left: leg_data = leg_data('BL', 3)
    back_right: leg_data = leg_data('BR', 4)


class WalkingController():
    def __init__(self, gait_type='trot', phase=[0, PI, PI, 0]):
        self._phase = robot_data(front_right=phase[0], front_left=phase[1], back_right=phase[2], back_left=phase[3])
        self.front_left = leg_data('FL', 1)
        self.front_right = leg_data('FR', 2)
        self.back_left = leg_data('BL', 3)
        self.back_right = leg_data('BR', 4)
        self.gait_type = gait_type

        self.link_lengths_stochlite = [0.096, 0.146, 0.172]

        self.leg_name_to_sol_branch_HyQ = {'FL': 0, 'FR': 0, 'BL': 1, 'BR': 1}
        self.leg_name_to_dir_Laikago = {'FL': 1, 'FR': -1, 'BL': 1, 'BR': -1}
        self.leg_name_to_sol_branch_Laikago = {'FL': 0, 'FR': 0, 'BL': 0, 'BR': 0}
        # self.leg_name_to_sol_branch_Stochlite = {'FL': 1, 'FR': -1, 'BL': 1, 'BR': -1}

        self.robot_width = 0.192
        self.robot_length = 0.334
        self.stochlite_kin = StochliteKinematics()

    def update_leg_theta(self, theta):
        """ Depending on the gait, the theta for every leg is calculated"""

        def constrain_theta(theta):
            theta = np.fmod(theta, 2 * no_of_points)
            if (theta < 0):
                theta = theta + 2 * no_of_points
            return theta

        self.front_right.theta = constrain_theta(theta + self._phase.front_right)
        self.front_left.theta = constrain_theta(theta + self._phase.front_left)
        self.back_right.theta = constrain_theta(theta + self._phase.back_right)
        self.back_left.theta = constrain_theta(theta + self._phase.back_left)

    def initialize_elipse_shift(self, x_shift, y_shift, z_shift):
        '''
        Initialize desired X, Y, Z offsets of elliptical trajectory for each leg
        '''

        self.front_left.x_shift = x_shift[0]
        self.front_right.x_shift = x_shift[1]
        self.back_left.x_shift = x_shift[2]
        self.back_right.x_shift = x_shift[3]
        
        self.front_left.y_shift = y_shift[0]
        self.front_right.y_shift = y_shift[1]
        self.back_left.y_shift = y_shift[2]
        self.back_right.y_shift = y_shift[3]
        
        self.front_left.z_shift = z_shift[0]
        self.front_right.z_shift = z_shift[1]
        self.back_left.z_shift = z_shift[2]
        self.back_right.z_shift = z_shift[3]

    def initialize_leg_state(self, theta, action):
        '''
        Initialize all the parameters of the leg trajectories
        Args:
            theta  : trajectory cycle parameter theta
            action : trajectory modulation parameters predicted by the policy
        Ret:
            legs   : namedtuple('legs', 'front_right front_left back_right back_left')
        '''
        Legs = namedtuple('legs', 'front_left front_right back_left back_right')
        legs = Legs(front_left=self.front_left, front_right=self.front_right,
                    back_left=self.back_left,  back_right=self.back_right)

        self.update_leg_theta(theta)

        #for stochlite the order is fl,fr,bl,br after an action transform is done in the env files 

        leg_sl = action[:4]  # fr fl br bl
        leg_phi = action[4:8]  # fr fl br bl

        self._update_leg_phi_val(leg_phi)
        self._update_leg_step_length_val(leg_sl)
        
        # Action changed
        # action[:4] -> step_length fl fr bl br
        # action[4:8] -> steer angle 
        # action[8:12] -> x_shift fl fr bl br
        # action[12:16] -> y_shift fl fr bl br
        # action[16:20] -> z_shift fl fr bl br

        self.initialize_elipse_shift(action[8:12], action[12:16], action[16:20])

        return legs
    
    def run_elliptical_Traj_Stochlite(self, theta, action):
        '''
        Semi-elliptical trajectory controller
        Args:
            theta  : trajectory cycle parameter theta
            action : trajectory modulation parameters predicted by the policy
        Ret:
            leg_motor_angles : list of motors positions for the desired action [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
            Note: we are using the right hand rule for the conventions of the leg which is - x->front, y->left, z->up
        '''
        legs = self.initialize_leg_state(theta, action)

        z_center = -0.25 # changed, initial -0.28, changed wrt reset angles
        foot_clearance = 0.06

        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * PI
            leg.r = leg.step_length / 2

            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) + leg.x_shift # negate this equation if the bot walks backwards
                if leg_theta > PI: # theta taken from +x, CW # Flip this sigh if the trajectory is mirrored
                    flag = 0 #z-coordinate of ellipse, during stance_phase of walking
                else:
                    flag = 1 #z-coordinate of ellipse, during swing_phase of walking
                z = foot_clearance * np.sin(leg_theta) * flag + z_center + leg.z_shift

            leg.x, leg.y, leg.z = np.array(
                [[np.cos(leg.phi), -np.sin(leg.phi), 0], [np.sin(leg.phi), np.cos(leg.phi), 0], [0, 0, 1]]) @ np.array(
                [x, 0, z]) # rotating about z by steer_angle phi, CCW

            if leg.name == "FR" or leg.name == "BR":
                leg.y = leg.y - self.link_lengths_stochlite[0] + leg.y_shift
            else:
                leg.y = leg.y + self.link_lengths_stochlite[0] + leg.y_shift # abd_link = 0.096, abd in x-z plane, not foot contact 
            
            # print("In walking controller")
            # print(leg.name, leg.x, leg.y, leg.z)

            # if leg.name == "FR" or leg.name == "BR":
            #     leg.x = -0.04
            #     leg.y = -0.056
            #     leg. z = -0.25
            # else:
            #     leg.x = -0.04
            #     leg.y = 0.056
            #     leg. z = -0.25

            branch = "<"
            _,[leg.motor_abduction, leg.motor_hip, leg.motor_knee] = self.stochlite_kin.inverseKinematics(leg.name, [leg.x, leg.y, leg.z], branch)
            # print(leg.motor_knee,leg.motor_hip,leg.motor_abduction)

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip,
                            legs.front_right.motor_knee,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip,
                            legs.back_right.motor_knee,
                            legs.front_left.motor_abduction, legs.front_right.motor_abduction,
                            legs.back_left.motor_abduction, legs.back_right.motor_abduction]
        
        # leg_motor_angles = np.zeros(12) 
        # print("Angles")
        # print(leg_motor_angles)

        return leg_motor_angles

    def _update_leg_phi_val(self, leg_phi):
        '''
        Args:
             leg_phi : steering angles for each leg trajectories
        '''
        
        self.front_left.phi = leg_phi[0]
        self.front_right.phi = leg_phi[1]
        self.back_left.phi = leg_phi[2]
        self.back_right.phi = leg_phi[3]
        
    def _update_leg_step_length_val(self, step_length):
        '''
        Args:
            step_length : step length of each leg trajectories
        '''
        
        self.front_left.step_length = step_length[0]
        self.front_right.step_length = step_length[1]
        self.back_left.step_length = step_length[2]
        self.back_right.step_length = step_length[3]
        

    '''
    Conventions for all robots below this needs to be changed.

    Conventions used in StochLite -> Right Hand Rule, x: forwards, z: upwards

    def run_elliptical_Traj_Stoch2(self, theta, action):
        ''
        Semi-elliptical trajectory controller
        Args:
            theta  : trajectory cycle parameter theta
            action : trajectory modulation parameters predicted by the policy
        Ret:
            leg_motor_angles : list of motors positions for the desired action [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
        ''
        legs = self.initialize_leg_state(theta, action)

        y_center = -0.244
        foot_clearance = 0.06

        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * PI
            leg.r = leg.step_length / 2

            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) + leg.x_shift
                if leg_theta > PI:
                    flag = 0
                else:
                    flag = 1
                y = foot_clearance * np.sin(leg_theta) * flag + y_center + leg.y_shift

            leg.x, leg.y, leg.z = np.array(
                [[np.cos(leg.phi), 0, np.sin(leg.phi)], [0, 1, 0], [-np.sin(leg.phi), 0, np.cos(leg.phi)]]) @ np.array(
                [x, y, 0])
            leg.z = leg.z + leg.z_shift

            leg.motor_knee, leg.motor_hip, leg.motor_abduction = self.Stoch2_Kin.inverseKinematics(leg.x, leg.y, leg.z)
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Stoch2[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Stoch2[1]

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip,
                            legs.front_right.motor_knee,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip,
                            legs.back_right.motor_knee,
                            legs.front_left.motor_abduction, legs.front_right.motor_abduction,
                            legs.back_left.motor_abduction, legs.back_right.motor_abduction]

        return leg_motor_angles

    def run_elliptical_Traj_HyQ(self, theta, action):
        ''
        Semi-elliptical trajectory controller
        Args:
            theta  : trajectory cycle parameter theta
            action : trajectory modulation parameters predicted by the policy
        Ret:
            leg_motor_angles : list of motors positions for the desired action [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
        ''
        legs = self.initialize_leg_state(theta, action)

        y_center = -0.7
        foot_clearance = 0.12

        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * PI
            leg.r = leg.step_length / 2

            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) - leg.x_shift
                if leg_theta > PI:
                    flag = 0
                else:
                    flag = 1
                y = foot_clearance * np.sin(leg_theta) * flag + y_center - leg.y_shift

            leg.x, leg.y, leg.z = np.array(
                [[np.cos(leg.phi), 0, np.sin(leg.phi)], 
                
                [0, 1, 0], [-np.sin(leg.phi), 0, np.cos(leg.phi)]]) @ np.array(
                [x, y, 0])
            leg.z = leg.z - leg.z_shift
            leg.z = -1 * leg.z

            leg.motor_knee, leg.motor_hip, leg.motor_abduction = self.Hyq_Kin.inverseKinematics(leg.x, leg.y, leg.z,
                                                                                                self.leg_name_to_sol_branch_HyQ[
                                                                                                    leg.name])
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_HYQ[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_HYQ[1]
            leg.motor_abduction = -1 * leg.motor_abduction

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip,
                            legs.front_right.motor_knee, legs.back_left.motor_hip, legs.back_left.motor_knee,
                            legs.back_right.motor_hip, legs.back_right.motor_knee, legs.front_left.motor_abduction,
                            legs.front_right.motor_abduction, legs.back_left.motor_abduction,
                            legs.back_right.motor_abduction]

        return leg_motor_angles

    def run_elliptical_Traj_Laikago(self, theta, action):
        ''
        Semi-elliptical trajectory controller
        Args:
            theta  : trajectory cycle parameter theta
            action : trajectory modulation parameters predicted by the policy
        Ret:
            leg_motor_angles : list of motors positions for the desired action [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
        ''
        legs = self.initialize_leg_state(theta, action)

        y_center = -0.35
        foot_clearance = 0.1

        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * PI
            leg.r = leg.step_length / 2

            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) + leg.x_shift
                if leg_theta > PI:
                    flag = 0
                else:
                    flag = 1
                y = foot_clearance * np.sin(leg_theta) * flag + y_center + leg.y_shift

            leg.x, leg.y, leg.z = np.array(
                [[np.cos(leg.phi), 0, np.sin(leg.phi)], [0, 1, 0], [-np.sin(leg.phi), 0, np.cos(leg.phi)]]) @ np.array(
                [x, y, 0])

            leg.z = leg.z + leg.z_shift

            if leg.name == "fl" or leg.name == "bl":
                leg.z = -leg.z
        
            leg.motor_knee, leg.motor_hip, leg.motor_abduction = self.Laikago_Kin.inverseKinematics(leg.x, leg.y, leg.z,
                                                                                                    self.leg_name_to_sol_branch_Laikago[
                                                                                                        leg.name])

            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Laikago[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Laikago[1]
            leg.motor_abduction = leg.motor_abduction * self.leg_name_to_dir_Laikago[leg.name]
            leg.motor_abduction = leg.motor_abduction + 0.07


        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_left.motor_abduction,
                            legs.back_right.motor_hip, legs.back_right.motor_knee, legs.back_right.motor_abduction,
                            legs.front_right.motor_hip, legs.front_right.motor_knee, legs.front_right.motor_abduction,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_left.motor_abduction]

        return leg_motor_angles

    '''

def constrain_abduction(angle):
    '''
    constrain abduction command with respect to the kinematic limits of the abduction joint
    '''
    if (angle < 0):
        angle = 0
    elif (angle > 0.35):
        angle = 0.35
    return angle


if (__name__ == "__main__"):
    # walkcon = WalkingController(phase=[PI, 0, 0, PI])
    walkcon = WalkingController(phase=[0, PI, PI, 0])
    # walkcon._update_leg_step_length(0.068 * 2, 0.4)
    # walkcon._update_leg_phi(0.4)

