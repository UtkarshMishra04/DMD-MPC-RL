# ### Joystick Class
# Written by Tejas Rane (May, 2021)
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Utilities for simulating Joystick commands from the user.
'''

import random
import numpy as np
import matplotlib.pyplot as plt

class Joystick():
    def __init__(self, episode_length, curilearn):
        self.total_eps = episode_length
        self.commands = []

        if curilearn:
            # Direction adn Command Randomizer, based on Curiculum learning
            self.rand_vx = -random.randint(-1, 1) # default: 1
            self.rand_vy = -random.randint(-1, 1) # default: 0
            self.rand_wz = -random.randint(-1, 1) # default: 0
            # print('randomizer, ', self.rand_vx, self.rand_vy, self.rand_wz)
        else:
            self.rand_vx = 1
            self.rand_vy = 0
            self.rand_wz = 0

    def step(self):
        '''
        Function to generate a Step command input.
        Command is 0.0 for the 20% of the episode length, command is 1.0 for the remainder.
        '''

        for i in range(self.total_eps):
            ratio = i/self.total_eps
            if ratio < 0.2:
                cmd = 0.0
            else:
                cmd = 1.0

            self.commands.append([self.rand_vx*cmd, self.rand_vy*cmd, self.rand_wz*cmd])
        
        return self.commands
 
    def mod_step(self):
        '''
        Function to generate a Modified Step command input.
        Command is 0.0 for the 20% of the episode length, command is 0.5 from 20% to 80%, command is 1.0 for the remainder.
        '''

        for i in range(self.total_eps):
            ratio = i/self.total_eps
            if ratio < 0.2:
                cmd = 0.0
            elif ratio < 0.8:
                cmd = 0.5
            else:
                cmd = 1.0

            self.commands.append([self.rand_vx*cmd, self.rand_vy*cmd, self.rand_wz*cmd])
        
        return self.commands

    def ramp(self):
        '''
        Function to generate a Ramp command input.
        Command is the ratio of number of steps completed to the total episode length, hence generating a ramp function.
        '''

        for i in range (self.total_eps):
            ratio = i/self.total_eps
            cmd = ratio

            self.commands.append([self.rand_vx*cmd, self.rand_vy*cmd, self.rand_wz*cmd])

        return self.commands

    def mod_ramp(self):
        '''
        Function to generate a Modified Ramp command input.
        Command is 0.0 for the 20% of the episode length, command is a ramp function from 20% to 80%, command is 1.0 for the remainder.
        '''

        ramp_start = (int)(0.2 * self.total_eps)
        ramp_end = (int)(0.80 * self.total_eps)

        for i in range (ramp_start):
            cmd = 0.0
            self.commands.append([self.rand_vx*cmd, self.rand_vy*cmd, self.rand_wz*cmd])

        for i in range (ramp_start, ramp_end):
            ratio = i/ramp_end
            cmd = ratio

            self.commands.append([self.rand_vx*cmd, self.rand_vy*cmd, self.rand_wz*cmd])
        
        for i in range (ramp_end, self.total_eps):
            cmd = 1.0
            self.commands.append([self.rand_vx*cmd, self.rand_vy*cmd, self.rand_wz*cmd])

        return self.commands

    def sinusoid(self):
        '''
        Function to generate a Sinusoidal command input.
        Command is sinusoid generated with respect to the number of steps completed (goes from 0 to pi).
        '''

        for i in range (self.total_eps):
            i = i * np.pi/self.total_eps
            cmd = np.sin(i)
            self.commands.append([self.rand_vx*cmd, self.rand_vy*cmd, self.rand_wz*cmd])

        return self.commands

    def step_in_place(self):
        '''
        Function to generate a Step in Place command input.
        Command is 0.0 for the entire episode length.

        Note:- This Function is not required as such, as it just generates a 0 command input.
        This is included here to just keep all command generation functions together.
        '''

        for i in range (self.total_eps):
            self.commands.append([0.0, 0.0, 0.0])

        return self.commands

    def randomize(self):
        '''
        Function to randomize a command input.
        '''

        ch = random.randint(1, 6)
        if ch == 1:
            return self.step()
        elif ch == 2:
            return self.mod_step()
        elif ch == 3:
            return self.ramp()
        elif ch == 4:
            return self.mod_ramp()
        elif ch == 5:
            return self.sinusoid()
        elif ch == 6:
            return self.step_in_place()

    def get_commands(self, choice):
        '''
        Function to choose input function and generate commands based on the user input.

        Note:- can add dynamic functionality to dynamically change %ages in in different input signal functions.
        '''

        func = {
            'st': self.step,
            "mst": self.mod_step,
            "r": self.ramp,
            "mr": self.mod_ramp,
            "sin": self.sinusoid,
            "sip": self.step_in_place,
            "rand": self.randomize
        }

        chosen_func = func.get(choice)

        return chosen_func()

if __name__ == '__main__':

    js = Joystick(1000)
    s = 'mst'
    cmd_vel = js.get_commands(s)
    x_vel = [p[0] for p in cmd_vel]
    plt.plot(x_vel)
    plt.show()
    