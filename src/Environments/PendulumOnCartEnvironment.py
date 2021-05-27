
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
            problem_parameters=None,
            custom_reward_function=None,
            custom_termination_function=None,
            action_space=[-10, 0, 10],
            lamb=0.1):
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

        super().__init__(problem, action_space, step_size,"PendulumOnCart")

        # Punishment for termination
        # TODO: Not used - use or remove?
        self.termination_reward = -1
        self.lamb = lamb

        if custom_reward_function is not None:
            self._reward = custom_reward_function
        else:
            self._reward = self._default_reward_function

        if custom_termination_function is not None:
            self._terminated = custom_termination_function
        else:
            self._terminated = self._default_terminated_function

    def reward(self, state, t):
        """
        """
        return self._reward(state, t)

    def terminated(self, state, t):
        """
        TODO
        """
        return self._terminated(state, t)

    def reset(self, random=False,):
        """
        Resets the simulation to its initial conditions
        By setting random to True, new initial conditions are generated randomly
        """
        if random:
            #initial_state = [np.random.uniform(-0.05,0.05) for _ in range(self.state_size)]
            initial_state = np.zeros(self.state_size)
            initial_state[2] = np.random.uniform(-self.lamb, self.lamb)
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
        self.problem.animate(save=True, filename=filename, title=title, hide=True, max_angle=10)

    def _default_reward_function(self, state, t):
        """
        Computes a reward based on the current state of the system.

        TODO: Add summary
        """
        # 20 degrees
        max_angle = 20*np.pi/180

        # TODO: Should this be a model constraint?
        max_cart_pos = 2.5

        x, xdot, theta, thetadot = state
        r_angle = (max_angle - abs(theta))/max_angle
        r_pos = (max_cart_pos - abs(x))/max_cart_pos
        reward = np.cos(theta*10) # + r_pos

        return reward

    def _default_terminated_function(self, state, t):
        """
        Checks whether the system has entered a termination state.

        TODO: Add summary
        """
        # 20 degrees
        max_angle = 20*np.pi/180

        # TODO: Should this be a model constraint?
        max_cart_pos = 2.5

        #Calculate Termination
        x, xdot, theta, thetadot = state
        return abs(theta) > max_angle or abs(x) > max_cart_pos

if __name__ == "__main__":
    from PendulumOnCartSimulator import PendulumOnCartSimulator

    def policy(x):
        """
        Dummy function
        Returns an input for given states and time
        """
        return np.random.choice([0, 1, 2])

    env = PendulumOnCartEnvironment()
    state = env.get_state()
    n_steps = 200
    print(f"Initial State = {state}")
    for i in range(n_steps):
        action = policy(state)
        next_state = env.step(action)
        reward = env.reward(next_state, i * env.step_size)
        terminated = env.terminated(next_state, i * env.step_size)
        print(f"Step = {i}, Action = {action}, Reward = {reward}, Terminated = {terminated}")
        state = next_state

    env.animate()
