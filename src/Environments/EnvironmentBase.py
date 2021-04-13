###################################
#
#   Base for Controller Classes
#
###################################

import numpy as np
import random
from abc import ABC, abstractmethod

class EnvironmentBase(ABC):
    """
    Describes an environment.

    TODO: Complete summary.
    """

    def __init__(self, problem, action_space, step_size, name):
        self.problem = problem
        self.state_size = len(problem.initial_state)
        self.action_space = action_space
        self.step_size = step_size
        self.name = name

    def step(self, action_index):
        """
        Send input to the simulator and return new state
        """
        action_value = self.action_space[action_index]
        return self.problem.step(self.step_size, action_value)

    def solve(self, t, controller=None):
        """
        """
        return self.problem.solve(t, controller=controller)

    def get_state(self):
        """
        Returns current state of the simulation
        """
        return self.problem.get_current_state()

    def get_action_space(self):
        """
        """
        return self.action_space

    def get_random_action(self):
        """
        TODO: Complete summary.
        """
        return random.choice(range(len(self.action_space)))

    @abstractmethod
    def reset(self, random=False):
        """
        Resets simulation
        """
        pass

    @abstractmethod
    def reward(self, state, t=None):
        """
        Calculates reward for current condition
        """
        pass

    @abstractmethod
    def terminated(self, state, t=None):
        """
        Calculates termination for state
        """
        pass

    @abstractmethod
    def save(self, episode):
        pass

class SimpleProblem:
    initial_state = [1, 2]

    def get_current_state(self):
        return self.initial_state

    def step(self, x, t):
        pass

class SimpleEnvironment(EnvironmentBase):
    """
    Simple Test
    """

    def __init__(self, step_size=0.1):
        problem = SimpleProblem()
        super().__init__(SimpleProblem(), action_space=[1, 2], step_size=step_size)

        self.lim = 10
        self.maxpos = 2

    def reset(self):
        pass

    def reward(self, state):
        return 1

    def terminated(self, state):
        return False

if __name__ == "__main__":
    env = SimpleEnvironment(0.1)
    def policy(x=None,t=None):
        return np.random.choice([0, 1])

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
