# ### Trajectory Generator
# Written by Tejas Rane (May, 2021)
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for realizing walking controllers.

Action space is defined as:
action[:4] = x_shift fl fr bl br
action[4:8] = y_shift fl fr bl br
action[8:12] = z_shift fl fr bl br
action[12] = linear x velocity of robot (lin_x_vel)
action[13] = linear y velocity of robot (lin_y_vel)
action[14] = angular velocity of robot about z (ang_z_vel)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import dataclass
from collections import namedtuple
from envs.environments.stoch_env.utils.ik_class import StochliteKinematics
import numpy as np
import matplotlib.pyplot as plt

PI = np.pi
no_of_points = 100

@dataclass
class leg_data:
    name: str
    ID: int
    theta: float = 0.0
    prev_motor_hip: float = 0.0
    prev_motor_knee: float = 0.0
    prev_motor_abd: float = 0.0
    motor_hip: float = 0.0
    motor_knee: float = 0.0
    motor_abd: float = 0.0
    prev_x: float = 0.0
    prev_y: float = 0.0
    prev_z: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    x_shift: float = 0.0
    y_shift: float = 0.0
    z_shift: float = 0.0


@dataclass
class robot_data:
    front_left: leg_data = leg_data('FL', 1)
    front_right: leg_data = leg_data('FR', 2)
    back_left: leg_data = leg_data('BL', 3)
    back_right: leg_data = leg_data('BR', 4)


class TrajectoryGenerator():
    def __init__(self, gait_type='trot', phase=[0, PI, PI, 0]):
        self.gait_type = gait_type
        self.phase = robot_data(front_left=phase[0], front_right=phase[1], back_left=phase[2], back_right=phase[3])
        self.frequency = 2.5
        self.omega = 2 * PI * self.frequency
        # self.omega = 2 * no_of_points * self.frequency
        self.theta = 0

        self.front_left = leg_data('FL', 1)
        self.front_right = leg_data('FR', 2)
        self.back_left = leg_data('BL', 3)
        self.back_right = leg_data('BR', 4)

        self.robot_width = 0.192
        self.robot_length = 0.334
        self.link_lengths_stochlite = [0.096, 0.146, 0.172]
        self.stochlite_kin = StochliteKinematics()

        self.foot_clearance = 0.08
        self.walking_height = -0.25
        self.swing_time = (1/self.frequency)/2 #0.25
        self.stance_time = self.swing_time
        self.max_linear_xvel = 0.5 #0.4, made zero for only ang vel # calculation is < 0.2 m steplength times the frequency 2.5 Hz
        self.max_linear_yvel = 0.25 #0.25, made zero for only ang vel # calculation is < 0.14 m times the frequency 2.5 Hz
        self.max_ang_vel = 2 #considering less than pi/2 steer angle # less than one complete rotation in one second

    def update_leg_theta(self, dt):
        '''
        Function to calculate the leg cycles of the trajectory, depending on the gait.
        '''

        def constrain_theta(theta):
            theta = np.fmod(theta, 2 * PI)
            if (theta < 0):
                theta = theta + 2 * PI
            return theta

        # def constrain_theta(theta):
        #     theta = np.fmod(theta, 2 * no_of_points)
        #     if (theta < 0):
        #         theta = theta + 2 * no_of_points
        #     return theta

        self.theta = constrain_theta(self.theta + self.omega * dt) 

        self.front_left.theta = constrain_theta(self.theta + self.phase.front_left) 
        self.front_right.theta = constrain_theta(self.theta + self.phase.front_right)
        self.back_left.theta = constrain_theta(self.theta + self.phase.back_left)
        self.back_right.theta = constrain_theta(self.theta + self.phase.back_right)

        #converting no of points notation to PI
        # self.front_left.theta = constrain_theta(self.theta + self.phase.front_left) * (2*PI)/(2*no_of_points) 
        # self.front_right.theta = constrain_theta(self.theta + self.phase.front_right) * (2*PI)/(2*no_of_points)
        # self.back_left.theta = constrain_theta(self.theta + self.phase.back_left) * (2*PI)/(2*no_of_points)
        # self.back_right.theta = constrain_theta(self.theta + self.phase.back_right) * (2*PI)/(2*no_of_points)

    def reset_theta(self):
        '''
        Function to reset the main cycle of the trajectory.
        '''

        self.theta = 0

    def initialize_traj_shift(self, x_shift, y_shift, z_shift):
        '''
        Initialize desired X, Y, Z offsets of trajectory for each leg
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

    def initialize_prev_motor_ang(self, prev_motor_angles):
        '''
        Initialize motor angles of previous time-step for each leg
        '''

        self.front_left.prev_motor_hip = prev_motor_angles[0]
        self.front_left.prev_motor_knee = prev_motor_angles[1]
        self.front_left.prev_motor_abd = prev_motor_angles[8]
     
        self.front_right.prev_motor_hip = prev_motor_angles[2]
        self.front_right.prev_motor_knee = prev_motor_angles[3]
        self.front_right.prev_motor_abd = prev_motor_angles[9]

        self.back_left.prev_motor_hip = prev_motor_angles[4]
        self.back_left.prev_motor_knee = prev_motor_angles[5]
        self.back_left.prev_motor_abd = prev_motor_angles[10]

        self.back_right.prev_motor_hip = prev_motor_angles[6]
        self.back_right.prev_motor_knee = prev_motor_angles[7]
        self.back_right.prev_motor_abd = prev_motor_angles[11]

    def foot_step_planner(self, leg, v_leg):
        '''
        Calculates the  absolute coordinate (wrt hip frame) where the foot should land at the beginning of the stance phase of the trajectory based on the 
        commanded velocities (either from joystick or augmented by the policy).
        Args:
            leg   : the leg for which the trajectory has to be calculated
            v_leg : the velocity vector for the leg (summation of linear and angular velocity components)
        Ret:
            x, y, z : absolute coordinate of the foot step.
        '''

        s = v_leg * self.stance_time/2
        x = s[0] + leg.x_shift
        y = s[1] + leg.y_shift
        z = s[2] + leg.z_shift

        return [x, y, z]

    def calculate_planar_traj(self, leg, v_x, v_y, w_z, dt):
        '''
        Calculates the x and y component of the trajectory based on the commanded velocities (either from joystick or augmented by the policy).
        Args:
            leg  : the leg for which the trajectory has to be calculated
            v_x  : linear velocity along +x
            v_y  : linear velocity along +y
            w_z  : angular velocity about +z
            dt   : control period
        Ret:
            x, y : calculated x and y coordinates of the trajectory.
        '''
        
        cmd_lvel = np.array([v_x, v_y, 0])
        cmd_avel = np.array([0, 0, w_z])
        v_leg = [0, 0, 0]
        next_step = [0, 0, 0]
        swing_vec = [0, 0, 0]

        if (leg.name == "FL"):
            leg_frame = [+self.robot_length/2, +self.robot_width/2, 0]
        elif (leg.name == "FR"):
            leg_frame = [+self.robot_length/2, -self.robot_width/2, 0]
        elif (leg.name == "BL"):
            leg_frame = [-self.robot_length/2, +self.robot_width/2, 0]
        elif (leg.name == "BR"):
            leg_frame = [-self.robot_length/2, -self.robot_width/2, 0]
        
        prev_foot_pos = np.array([leg.prev_x, leg.prev_y, 0])
        prev_r = prev_foot_pos + np.array(leg_frame)

        v_lcomp = cmd_lvel
        v_acomp = np.cross(cmd_avel, prev_r)
        v_leg = v_lcomp + v_acomp

        if leg.theta > PI: # theta taken from +x, CW # Flip this sign if the trajectory is mirrored
            flag = -1 #during stance_phase of walking, leg moves backwards to push body forward
            dr = v_leg * dt * flag
            r = prev_r + dr - np.array(leg_frame)
        else:
            flag = 1 #during swing_phase of walking, leg moves forward to next step
            next_step = self.foot_step_planner(leg, v_leg) + np.array(leg_frame)
            swing_vec = next_step - prev_r
            time_left = (PI - leg.theta)/PI * self.swing_time
            if time_left == 0:
                dr = 0 
            else:
                dr = swing_vec/time_left * dt * flag
            r = prev_r + dr - np.array(leg_frame)


        x = r[0]
        y = r[1]

        return [x, y]

    def cspline_coeff(self, z0, z1, d0, d1, t):
        '''
        Generates coefficients for the sections of the cubic spline based on the boundary conditions
        Equation -> z = coefft[3]*t**3 + coefft[2]*t**2 + coefft[1]*t**1 + coefft[0]*t**0
        Args:
            z0 : initial z
            z1 : final z
            d0 : initial dz/dtheta
            d1 : final dz/dtheta
            t  : domain of the section [(0, z0) to (t, z1), initial and final control points]
        Ret:
            coefft : list of coefficients for that section of cubic spline.
        '''

        coefft = [0]*4
        coefft[0] = z0
        coefft[1] = d0
        w0 = z1 - z0 - d0*t
        w1 = d1 - d0
        coefft[2] = -1*(-3*t**2*w0 + t**3*w1)/t**4
        coefft[3] = -1*(2*t*w0 - t**2*w1)/t**4
        return coefft

    def calculate_vert_comp(self, leg):
        '''
        Calculates the z component of the trajectory. The function for the z component can be changed here.
        The z component calculation is kept independent as it is not affected by the velocity calculations. 
        Various functions can be used to smoothen out the foot impacts while walking.
        Args:
            leg : the leg for which the trajectory has to be calculated
        Ret:
            z   : calculated z component of the trajectory.
        '''

        if leg.theta > PI: # theta taken from +x, CW # Flip this sigh if the trajectory is mirrored
            flag = 0 #z-coordinate of trajectory, during stance_phase of walking
        else:
            flag = 1 #z-coordinate of trajectory, during swing_phase of walking

        # Sine function
        # z = self.foot_clearance * np.sin(leg.theta) * flag + self.walking_height + leg.z_shift

        # Cubic Spline
        '''
        This cubic spline is defined by 5 control points (n=4). Each control point is (theta, z) ie (theta_0, z_0) to (theta_n, z_n) n+1 control points
        The assumed cubic spline is of the type:
        z = coefft_n[3]*(t-t_n)**3 + coefft_n[2]*(t-t_n)**2 + coefft_n[1]*(t-t_n)**1 + coefft_n[0]*(t-t_n)**0 {0<=n<=3}
        where, n denotes each section of the cubic spline, governed by the nth-index control point 
        '''
        # theta = [0, PI/4, PI/2, 3*PI/4, PI]
        z = [0.0, 3*self.foot_clearance/4, self.foot_clearance, self.foot_clearance/2, 0.0]
        d = [0.1, 0.05, 0.0, -0.1, 0.0] # dz/dtheta at each control point
        t_vec = []
        coeffts = []

        if(leg.theta < PI/4):
            idx = 0
            coeffts = self.cspline_coeff(z[idx], z[idx+1], d[idx], d[idx+1], PI/4)
            t_vec = [leg.theta**i for i in range(4)]
        elif(leg.theta >= PI/4 and leg.theta < PI/2):
            idx = 1
            coeffts = self.cspline_coeff(z[idx], z[idx+1], d[idx], d[idx+1], PI/4)
            t_vec = [(leg.theta - PI/4)**i for i in range(4)]
        elif(leg.theta >= PI/2 and leg.theta < 3*PI/4):
            idx = 2
            coeffts = self.cspline_coeff(z[idx], z[idx+1], d[idx], d[idx+1], PI/4)
            t_vec = [(leg.theta - 2*PI/4)**i for i in range(4)]
        elif(leg.theta >= 3*PI/4 and leg.theta < PI):
            idx = 3
            coeffts = self.cspline_coeff(z[idx], z[idx+1], d[idx], d[idx+1], PI/4)
            t_vec = [(leg.theta - 3*PI/4)**i for i in range(4)]
        t_vec = np.array(t_vec)
        coeffts = np.array(coeffts)
        val = coeffts.dot(t_vec)

        z = val * flag + self.walking_height + leg.z_shift

        return z

    def safety_check(self, x, y, z):
        '''
        Performs a safety check over the planned foot pos according to the kinematic limits of the leg. 
        Calculates the corrected, safe foot pos, if required.
        Args:
            x, y, z : planned foot pos according to the cmd vel and the trajectory.
        Ret:
            x, y, z : corrected, safe foot pos if the planned foot pos was outside 80% of the workspace (extra safety), or the planned foot pos.
        '''

        mag = np.sqrt((x**2 + y**2 + z**2)) # Magnitude of planned foot pos vector from hip (leg origin)
        if mag == 0:
            print('Invalid Point')

        # Safety Calculations
        r = np.sqrt((self.link_lengths_stochlite[0]**2 + (self.link_lengths_stochlite[1] + self.link_lengths_stochlite[2])**2)) # max radius of workspace of leg, equation of sphere
        
        if (y**2 + z**2) < self.link_lengths_stochlite[0]**2:
            y = self.link_lengths_stochlite[0] * (y/mag)
            z = self.link_lengths_stochlite[0] * (z/mag)
        
        if (x**2 + y**2 + z**2) <= (0.9*r)**2:
            # within 90% of max radius, extra factor of safety
            return [x, y, z]
        else:
            #print('safety')
            x = (0.9*r) * (x/mag)
            y = (0.9*r) * (y/mag)
            z = (0.9*r) * (z/mag)
            return [x, y, z] 

    def initialize_leg_state(self, action, prev_motor_angles, dt):
        '''
        Initialize all the parameters of the leg trajectories
        Args:
            action            : trajectory modulation parameters predicted by the policy
            prev_motor_angles : joint encoder values for the previous control step
            dt                : control period 
        Ret:
            legs : namedtuple('legs', 'front_right front_left back_right back_left')
        '''

        Legs = namedtuple('legs', 'front_left front_right back_left back_right')
        legs = Legs(front_left=self.front_left, front_right=self.front_right,
                    back_left=self.back_left,  back_right=self.back_right)

        self.update_leg_theta(dt)

        self.initialize_traj_shift(action[:4], action[4:8], action[8:12])
        self.initialize_prev_motor_ang(prev_motor_angles)

        return legs

    def generate_trajectory(self, action, prev_motor_angles, dt):
        '''
        Velocity based trajectory generator. The controller assumes a default trot gait. 
        Args:
            action : trajectory modulation parameters predicted by the policy
            prev_motor_angles : joint encoder values for the previous control step
            dt                : control period
        Ret:
            leg_motor_angles : list of motors positions for the desired action [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
            Note: we are using the right hand rule for the conventions of the leg which is - x->front, y->left, z->up
        '''

        legs = self.initialize_leg_state(action, prev_motor_angles, dt)
        lin_vel_x = action[12] * self.max_linear_xvel
        lin_vel_y = action[13] * self.max_linear_yvel
        ang_vel_z = action[14] * self.max_ang_vel

        for leg in legs:

            # [leg.prev_x, leg.prev_y, leg.prev_z] = self.stochlite_kin.forwardKinematics(leg.name, [leg.prev_motor_abd, leg.prev_motor_hip, leg.prev_motor_knee]) # Closed-loop feedback from FK
            [leg.prev_x, leg.prev_y, leg.prev_z] = [leg.x, leg.y, leg.z] # Open-loop
            [leg.x, leg.y] = self.calculate_planar_traj(leg, lin_vel_x, lin_vel_y, ang_vel_z, dt)
            leg.z = self.calculate_vert_comp(leg)
            [leg.x, leg.y, leg.z] = self.safety_check(leg.x, leg.y, leg.z)

            branch = ">"
            _,[leg.motor_abd, leg.motor_hip, leg.motor_knee] = self.stochlite_kin.inverseKinematics(leg.name, [leg.x, leg.y, leg.z], branch)

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip,
                            legs.front_right.motor_knee,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip,
                            legs.back_right.motor_knee,
                            legs.front_left.motor_abd, legs.front_right.motor_abd,
                            legs.back_left.motor_abd, legs.back_right.motor_abd]

        # Plotting in base frame
        # fl_foot_pos = [legs.front_left.x + self.robot_length/2, legs.front_left.y + self.robot_width/2, legs.front_left.z]
        # fr_foot_pos = [legs.front_right.x + self.robot_length/2, legs.front_right.y - self.robot_width/2, legs.front_right.z]
        # bl_foot_pos = [legs.back_left.x - self.robot_length/2, legs.back_left.y + self.robot_width/2, legs.back_left.z]
        # br_foot_pos = [legs.back_right.x - self.robot_length/2, legs.back_right.y - self.robot_width/2, legs.back_right.z]

        # Plotting in leg frame
        # fl_foot_pos = [legs.front_left.x, legs.front_left.y, legs.front_left.z]
        # fr_foot_pos = [legs.front_right.x, legs.front_right.y, legs.front_right.z]
        # bl_foot_pos = [legs.back_left.x, legs.back_left.y, legs.back_left.z]
        # br_foot_pos = [legs.back_right.x, legs.back_right.y, legs.back_right.z]
        
        return leg_motor_angles #, fl_foot_pos + fr_foot_pos + bl_foot_pos + br_foot_pos

if __name__ == '__main__':
    '''
    This script can be run independently to plot the generated trajectories in either the leg frame or the robot base frame.

    Note: To run this file independently, copy-paste the /SlopedTerrainLinearPolicy/utils folder from the
    /SlopedTerrainLinearPolicy folder to /SlopedTerrainLinearPolicy/gym_sloped_terrains/envs  
    '''

    trajgen = TrajectoryGenerator()
    action = np.array([0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0])
    
    plotdata = []
    dt = 0.01
    prev_motor_angles = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ax = plt.axes(projection='3d')

    for i in range(200):
        prev_motor_angles, foot_pos = trajgen.generate_trajectory(action, prev_motor_angles, dt)
        plotdata.append(foot_pos)

    x_fl = [p[0] for p in plotdata]
    y_fl = [p[1] for p in plotdata]
    z_fl = [p[2] for p in plotdata]
    x_fr = [p[3] for p in plotdata]
    y_fr = [p[4] for p in plotdata]
    z_fr = [p[5] for p in plotdata]
    x_bl = [p[6] for p in plotdata]
    y_bl = [p[7] for p in plotdata]
    z_bl = [p[8] for p in plotdata]
    x_br = [p[9] for p in plotdata]
    y_br = [p[10] for p in plotdata]
    z_br = [p[11] for p in plotdata]

    ax.plot3D(x_fl, y_fl, z_fl, 'red')
    ax.plot3D(x_fr, y_fr, z_fr, 'blue')
    ax.plot3D(x_bl, y_bl, z_bl, 'blue')
    ax.plot3D(x_br, y_br, z_br, 'red')

    plt.show()