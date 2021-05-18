###################################
#
#   Base for ODE problems.
#
###################################

import numpy as np
from scipy.integrate import odeint
from abc import ABC, abstractmethod

class OdeProblemBase(ABC):
    """
    Base class for ODE problems.
    """
    def __init__(self, initial_state):
        """
        Instantiates an OdeProblemBase.
        """
        self.initial_state = initial_state
        self.states = [initial_state]
        self.time = [0.0]
        self.u = [0.0]

    def get_initial_state(self):
        """
        Returns the initial state of the problem.

        TODO: Complete summary.
        """
        return self.initial_state

    def get_current_state(self):
        """
        Gets the current state of the system.

        TODO: Complete summary.
        """
        return self.states[-1]

    def get_states_and_time(self):
        """
        Gets the recorded states and time.

        TODO: Complete summary
        """
        return (self.states, self.time)


    def print(self):
        """
        Prints the model information.

        TODO: Complete summary.
        """
        print(f"Initial state: {self.initial_state}")

    def reset(self, new_initial_state=None):
        """
        Resets the ODE problem state.

        TODO: Complete summary.
        """
        if new_initial_state is None:
            new_initial_state = self.initial_state

        self.initial_state = new_initial_state
        self.states = [self.initial_state]
        self.time = [0.0]
        return self.initial_state

    def step(self, step_size: float, external_force=0.0):
        """
        Takes one step in time with given step size.

        TODO: Complete summary.
        """
        t = [0, step_size]
        current_state = self.get_current_state()
        new_state = odeint(self._rhs, current_state, t, args=(external_force,))[-1]

        # TODO: investigate performance compared to a regular list.
        # self.states = np.append(self.states, new_state, axis=0)
        self.states.append(new_state)
        self.time.append(self.time[-1] + step_size)
        self.u.append(external_force)

        return new_state

    def solve(self, t, controller=None):
        """
        Solves the problems for time t.

        TODO: Add description.

        TODO: Add error handling (input control -- what if timestep is negative etc.).

        TODO: Compute step size based on the timestamp (t is a np.linspace array)
              otherwise the solver might not end up at time t!
              alternatively require the user to supply the correct step size.

        TODO: Complete summary.
        """
        # Input control
        assert len(t) >= 2,  "Time array is smaller than 2. Input must have length >= 2"

        # TODO: This won't work as expected -- talk about this and fix.
        # timestep is not None by default so the code will not be invoked
        # and the default value 0.1 is most likely going to be used.
        # if timestep == None:
        timestep = t[1]-t[0]
        # assert timestep > 0, "Timestep must be larger than 0.0"
        # print(f"Solving system with timestep = {timestep} seconds")

        if controller is None:
            controller = lambda x, t : 0

        for i in t:
            current_state = self.get_current_state()
            external_force = controller(current_state, i)

            # Don't need to save the new step.
            # It's saved inside the step function.
            _ = self.step(timestep, external_force)

        #self.time = t

    def animate(self):
        """
        Animates the problem if possible and implemented by the inheritor.
        """
        pass

    def plot(self):
        """
        Plots the results.

        TODO: Needed? Or should the inheritor override?
        """
        pass

    @abstractmethod
    def _rhs(self, x, t, u):
        """
        defines the right hand side of the problem.
        """
        pass


class SimpleOde(OdeProblemBase):
    """
    Simple Test!
    """
    def __init__(self, initial_state):
        """
        """
        super().__init__(initial_state)

    def _rhs(self, x, t, u):
        theta = x[0]
        theta_dot = x[1]

        theta_dot_dot = -9.81 * np.sin(theta)
        dydx = [theta_dot, theta_dot_dot]
        return dydx

if __name__ == "__main__":
    # x0 = np.array([0.0, 0.0, 0.05, 0.0])
    # problem = OdeProblemBase(x0)
    # problem.print()

    x0 = np.array([0.0, 0.01])
    problem = SimpleOde(x0)
    problem.print()

    t = np.linspace(0, 10, 100)
    problem.solve(t)

    states, time = problem.get_states_and_time()
