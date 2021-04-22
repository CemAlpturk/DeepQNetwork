

"""
Handle the communication between the simulator and the agent
- Get an action from the agent and send it to the simulator
- Retrieve new state from simulator after each step and compute reward and termination status
"""

import numpy as np
from .EnvironmentBase import EnvironmentBase
from .DoublePendulumOnCartSimulator import DoublePendulumOnCartSimulator


class DoublePendulumOnCartEnvironment(EnvironmentBase):

    def __init__(
            self,
            step_size=0.01,
            problem_parameters=None,
            custom_reward_function=None,
            custom_termination_function=None,
            action_space=[-5, -1, 0, 1, 5]):
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
        initial_state = np.array([0, 0, 0, 0, 0, 0])
        problem = DoublePendulumOnCartSimulator(problem_parameters, initial_state)
        super().__init__(problem, action_space, step_size, "DoublePendulumOnCart")

        # Punishment for termination
        # TODO: Not used - should it be used or removed?
        self.termination_reward = -1

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

    def _default_reward_function(self, state, t):
        """
        Computes a reward based on the current state of the system.

        TODO: Add summary
        """
        # Calculate reward
        x, theta1, theta2, xdot, theta1dot, thteta2dot = state
        # r_angle1 = (self.max_angle - abs(theta1))/self.max_angle - 0.5
        r_angle2 = (self.max_angle - abs(theta2))/self.max_angle
        # r_pos = (self.max_cart_pos - abs(x))/self.max_cart_pos - 0.5
        #r_angle1 = np.cos(theta1*10)
        #r_angle2 = np.cos(theta2*10)
        r_pos = 0
        reward = r_angle2  + r_angle2 + r_pos

        return reward

    def _default_terminated_function(self, state, t):
        """
        Checks whether the system has entered a termination state.

        TODO: Add summary
        """
        max_angle = 20*np.pi/180
        max_cart_pos = 2.5

        #Calculate Termination
        x, theta1, theta2, xdot, theta1dot, thteta2dot = state
        return abs(theta1) > max_angle or  \
                abs(theta2) > max_angle or \
                abs(x) > max_cart_pos

    def reset(self, random=False):
        """
        Resets the simulation to its initial conditions
        By setting random to True, new initial conditions are generated randomly
        """
        max_angle = 5*np.pi/180
        if random:
            lamb = 0.1*max_angle
            #initial_state = [np.random.uniform(-0.05,0.05) for _ in range(self.state_size)]
            initial_state = np.zeros(self.state_size)
            initial_state[1:3] = np.random.uniform(-lamb,lamb,2)
            return self.problem.reset(initial_state)
        else:
            return self.problem.reset()

    def save(self, episode):
        filename = f"./results/DoublePendulumOnCart/Episode_{episode}.gif"
        title = f"Episode: {episode}"
        self.problem.animate(save=True, filename=filename, title=title, hide=True)

    def animate(self):
        self.problem.animate()

if __name__ == "__main__":
    def random_action_policy(x):
        """
        Dummy function
        Returns an input for given states and time
        """
        return np.random.choice([0, 1, 2, 3, 4])

    env = DoublePendulumOnCartEnvironment()
    state = env.get_state()
    n_steps = 200
    print(f"Initial State = {state}")
    for i in range(n_steps):
        action = random_action_policy(state)
        next_state = env.step(action)
        reward = env.reward(next_state, i * env.step_size)
        terminated = env.terminated(next_state, i * env.step_size)
        print(f"Step = {i}, Action = {action}, Reward = {reward}, Terminated = {terminated}")
        state = next_state

    env.animate()
