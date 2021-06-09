###############################
#
#   Double Pendulum on a Cart
#
###############################


from .OdeProblemBase import OdeProblemBase
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.lines as lines



class DoublePendulumOnCartSimulator(OdeProblemBase):
    """
    Double Pendulum on a Cart Problem
    https://www3.math.tu-berlin.de/Vorlesungen/SoSe12/Kontrolltheorie/matlab/inverted_pendulum.pdf
    """
    g = 9.81

    def __init__(self, parameters, initial_state):

        super().__init__(initial_state)

        self.m = parameters["cart_mass"]
        self.m1 = parameters["pendulum_1_mass"]
        self.m2 = parameters["pendulum_2_mass"]
        self.l1 = parameters["pendulum_1_length"]
        self.l2 = parameters["pendulum_2_length"]
        self.d1 = parameters["cart_friction"]
        self.d2 = parameters["pendulum_1_friction"]
        self.d3 = parameters["pendulum_2_friction"]

    def print(self):
        """
        Prints the model information.

        TODO: Complete summary.
        """
        print(f"Cart mass: {self.m}")
        print(f"Pendulum-1 mass: {self.m1}")
        print(f"Pendulum-2 mass: {self.m2}")
        print(f"Pendulum-1 length: {self.l1}")
        print(f"Pendulum-2 length: {self.l2}")
        print(f"Cart friction: {self.d1}")
        print(f"Pendulum-1 friction: {self.d2}")
        print(f"Pendulum-2 friction: {self.d3}")
        print(f"Initial conditions: {self.initial_state} \n\
    [\n \
        Cart x position: {self.initial_state[0]} \n \
        Pendulum-1 angle: {self.initial_state[1]}\n \
        Pendulum-2 angle: {self.initial_state[2]}\n \
        Cart velocity: {self.initial_state[3]}\n \
        Pendulum-1 angular velocity: {self.initial_state[4]}\n \
        Pendulum-2 angular velocity: {self.initial_state[5]}\n \
    ]")

    def _rhs(self, state, t, u):
        """
        Computes the dynamic equations for the double pendulum system
        and returns the derivative of the states
        """

        x, q1, q2, x_dot, q1_dot, q2_dot = state

        M = np.zeros((3,3))
        M[0][0] = self.m + self.m1 + self.m2
        M[0][1] = self.l1*(self.m1+self.m2)*np.cos(q1)
        M[0][2] = self.m2*self.l2*np.cos(q2)

        M[1][0] = self.l1*(self.m1+self.m2)*np.cos(q1)
        M[1][1] = self.l1**2 *(self.m1 + self.m2)
        M[1][2] = self.l1*self.l2*self.m2*np.cos(q1-q2)

        M[2][0] = self.l2*self.m2*np.cos(q2)
        M[2][1] = self.l1*self.l2*self.m2*np.cos(q1-q2)
        M[2][2] = self.l2**2 * self.m2

        F = np.zeros(3)


        F[0] = self.l1*(self.m1+self.m2)*q1_dot**2*np.sin(q1) + \
                self.m2*self.l2*q2_dot**2*np.sin(q2) - self.d1*x_dot + u

        F[1] = -self.l1*self.l2*self.m2*q2_dot**2*np.sin(q1-q2) + \
                self.g*(self.m1+self.m2)*self.l1*np.sin(q1) - self.d2*q1_dot

        F[2] = self.l1*self.l2*self.m2*q1_dot**2*np.sin(q1-q2) + \
                self.g*self.l2*self.m2*np.sin(q2) - self.d3*q2_dot

        tmp = np.linalg.inv(M).dot(F)

        dxdt = np.zeros(6)
        dxdt[0:3] = state[3:6]
        dxdt[3:6] = tmp

        return dxdt

# TODO: Remove?
if __name__ == "__main__":
    initial_state = np.array([0,0,0.01,0,0,0])
    parameters = {
            "cart_mass": 1.0,
            "pendulum_1_mass": 0.1,
            "pendulum_2_mass": 0.1,
            "pendulum_1_length": 1.0,
            "pendulum_2_length": 1.0,
            "cart_friction": 0.01,
            "pendulum_1_friction": 0.01,
            "pendulum_2_friction": 0.01
    }

    problem = DoublePendulumOnCartSimulator(parameters, initial_state)
    problem.print()
    state = problem.get_current_state()
    t = np.linspace(0,10,100)
    for i in t:
        results = problem.step(0.1)

    # problem.animate()
