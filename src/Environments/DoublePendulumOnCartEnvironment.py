

"""
Handle the communication between the simulator and the agent
- Get an action from the agent and send it to the simulator
- Retrieve new state from simulator after each step and compute reward and termination status
"""

import numpy as np
from EnvironmentBase import EnvironmentBase
from DoublePendulumOnCartSimulator import DoublePendulumOnCartSimulator


class DoublePendulumOnCartEnvironment(EnvironmentBase):

    def __init__(
            self,
            step_size=0.01,
            problem_parameters=None):
        """
        env: DoublePendulumOnCart object
        """
        if problem_parameters is None:
            problem_parameters = {
                    "cart_mass": 1.0,
                    "pendulum_1_mass": 0.1,
                    "pendulum_2_mass": 0.1,
                    "pendulum_1_length": 1.0,
                    "pendulum_2_length": 1.0
                }
        initial_state = np.array([0,0,0,0,0,0])
        problem = DoublePendulumOnCartSimulator(problem_parameters, initial_state)
        action_space = [-5,-1,0,1,5]
        super().__init__(problem, action_space, step_size, "DoublePendulumOnCart")

        # 12 degrees
        self.max_angle = 12*np.pi/180

        # TODO: Should this be a model constraint?
        self.max_cart_pos = 2.5

        # Punishment for termination
        self.termination_reward = -1


    def reward(self, state, t=None):
        """
        Computes a reward based on the current state of the system.

        TODO: Add summary
        """
        # Calculate reward
        x, theta1, theta2, xdot, theta1dot, thteta2dot = state
        r_angle1 = (self.max_angle - abs(theta1))/self.max_angle - 0.5
        r_angle2 = (self.max_angle - abs(theta2))/self.max_angle - 0.5
        r_pos = (self.max_cart_pos - abs(x))/self.max_cart_pos - 0.5
        reward = r_angle1 + r_angle2 + r_pos
        return reward

    def terminated(self, state, t=None):
        """
        Checks whether the system has entered a termination state.

        TODO: Add summary
        """
        #Calculate Termination
        x, theta1, theta2, xdot, theta1dot, thteta2dot = state
        return abs(theta1) > self.max_angle or  \
                abs(theta2) > self.max_angle or \
                abs(x) > self.max_cart_pos

    def reset(self, random=False):
        """
        Resets the simulation to its initial conditions
        By setting random to True, new initial conditions are generated randomly
        """
        if random:
            #initial_state = [np.random.uniform(-0.05,0.05) for _ in range(self.state_size)]
            initial_state = np.random.uniform(-0.05,0.05,self.state_size)
            return self.env.reset(initial_state)
        else:
            return self.env.reset()

    def save(self, episode):
        pass

    def animate(self):
        self.problem.animate()

if __name__ == "__main__":


    def random_action_policy(x):
        """
        Dummy function
        Returns an input for given states and time
        """
        return np.random.choice([0,1,2,3,4])

    env = DoublePendulumOnCartEnvironment(0.01)
    state = env.get_state()
    n_steps = 200
    print(f"Initial State = {state}")
    for i in range(n_steps):
        action = random_action_policy(state)
        next_state = env.step(action)
        reward = env.reward(next_state)
        terminated = env.terminated(next_state)
        print(f"Step = {i}, Action = {action}, Reward = {reward}, Terminated = {terminated}")
        state = next_state

    env.animate()
