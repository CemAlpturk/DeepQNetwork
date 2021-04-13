
"""
Handle the communication between the simulator and the agent
- Get an action from the agent and send it to the simulator
- Retrieve new state from simulator after each step and compute reward and termination status
"""
import os

import numpy as np

from .EnvironmentBase import EnvironmentBase
from .PendulumOnCartSimulator import PendulumOnCartSimulator

class PendulumOnCartEnvironment(EnvironmentBase):

    def __init__(
            self,
            step_size=0.1,
            problem_parameters=None):
        """
        env: PendulumOnCart object
        """
        if problem_parameters is None:
            problem_parameters = {
                "cart_mass": 1.0,
                "pendulum_mass": 0.1,
                "pendulum_length" : 1.0,
            }
        # TODO: May need to randomize the initial state.
        initial_state = np.array([0, 0, 0.01, 0])
        problem = PendulumOnCartSimulator(problem_parameters, initial_state)
        action_space = [-10, 0, 10]

        super().__init__(problem, action_space, step_size,"PendulumOnCart")

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
        x, xdot, theta, thetadot = state
        r_angle = (self.max_angle - abs(theta))/self.max_angle - 0.5
        r_pos = (self.max_cart_pos - abs(x))/self.max_cart_pos - 0.5
        reward = r_angle + r_pos
        return reward

    def terminated(self, state, t=None):
        """
        Checks whether the system has entered a termination state.

        TODO: Add summary
        """
        #Calculate Termination
        x, xdot, theta, thetadot = state
        return abs(theta) > self.max_angle or abs(x) > self.max_cart_pos


    def reset(self, random=False):
        """
        Resets the simulation to its initial conditions
        By setting random to True, new initial conditions are generated randomly
        """
        if random:
            #initial_state = [np.random.uniform(-0.05,0.05) for _ in range(self.state_size)]
            initial_state = np.random.uniform(-0.05,0.05, self.state_size)
            return self.problem.reset(initial_state)
        else:
            return self.problem.reset()

    def animate(self):
        """
        TODO: Complete summary.
        """
        self.problem.animate()

    def save(self, episode):
        """
        TODO: Complete Summary
        """
        filename = f"./results/PendulumOnCart/Episode_{episode}.gif"
        title = f"Episode: {episode}"
        self.problem.animate(save=True, filename=filename, title=title, hide=True)

if __name__ == "__main__":
    from pendulum_on_a_cart import PendulumOnCart

    def policy(x):
        """
        Dummy function
        Returns an input for given states and time
        """
        return np.random.choice([0, 1, 2])

    env = PendulumOnCartEnvironment(0.1)
    state = env.get_state()
    n_steps = 200
    print(f"Initial State = {state}")
    for i in range(n_steps):
        action = policy(state)
        next_state = env.step(action)
        reward = env.reward(next_state)
        terminated = env.terminated(next_state)
        print(f"Step = {i}, Action = {action}, Reward = {reward}, Terminated = {terminated}")
        state = next_state

    env.animate()
