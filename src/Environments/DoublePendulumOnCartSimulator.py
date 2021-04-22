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
                self.m2*self.l2*q2_dot**2*np.sin(q2) + u

        F[1] = -self.l1*self.l2*self.m2*q2_dot**2*np.sin(q1-q2) + \
                self.g*(self.m1+self.m2)*self.l1*np.sin(q1)

        F[2] = self.l1*self.l2*self.m2*q1_dot**2*np.sin(q1-q2) + \
                self.g*self.l2*self.m2*np.sin(q2)

        tmp = np.linalg.inv(M).dot(F)

        dxdt = np.zeros(6)
        dxdt[0:3] = state[3:6]
        dxdt[3:6] = tmp

        return dxdt

    def animate(self, save=False, filename=None, title="Double Pendulum on a Cart", hide=False):
        """
        TODO
        """
        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            aspect='equal',
            xlim = (-5,5),
            ylim = (-2,2),
            title = title
        )

        ax.grid()
        origin = [0.0, 0.0]

        # Create the cart
        cart_borders = [0.0 - 0.25, 0.0 - 0.25]
        cart = patches.Rectangle(
            cart_borders,
            0.5,
            0.5,
            facecolor='none',
            edgecolor='k'
        )

        # Pendulum arm
        theta01 = self.initial_state[1]
        initial_pendulum_bob1_location = [
            self.l1 * np.sin(theta01),
            self.l1 * np.cos(theta01)
        ]
        theta02 = self.initial_state[2]
        initial_pendulum_bob2_location = [
            initial_pendulum_bob1_location[0] + self.l2*np.sin(theta02),
            initial_pendulum_bob1_location[1] + self.l2*np.cos(theta02)
        ]

        pendulumArm1 = lines.Line2D(
            origin,
            initial_pendulum_bob1_location,
            color='r',
            marker='o'
        )

        pendulumArm2 = lines.Line2D(
            initial_pendulum_bob1_location,
            initial_pendulum_bob2_location,
            color='r',
            marker='o'
        )

        # Time
        time_text = ax.text(-2, 1.6, '', fontsize=15)
        def init():
            ax.add_patch(cart)
            ax.add_line(pendulumArm1)
            ax.add_line(pendulumArm2)
            time_text.set_text('Time 0.0')
            return cart, pendulumArm1, pendulumArm2, time_text

        def animate(i):

            cart_xpos, q1, q2, x_dot, q1_dot, q2_dot = self.states[i]

            # Cart
            cart_coordinate = [cart_xpos - 0.25, -0.25]
            cart.set_xy(cart_coordinate)

            # Pendulum
            x_pendulum_bob1 = cart_xpos + self.l1*np.sin(q1)
            y_pendulum_bob1 = self.l1*np.cos(q1)
            xpos1 = [cart_xpos, x_pendulum_bob1]
            ypos1 = [0.0, y_pendulum_bob1]

            x_pendulum_bob2 = x_pendulum_bob1 + self.l2*np.sin(q2)
            y_pendulum_bob2 = y_pendulum_bob1 + self.l2*np.cos(q2)
            xpos2 = [x_pendulum_bob1, x_pendulum_bob2]
            ypos2 = [y_pendulum_bob1, y_pendulum_bob2]

            pendulumArm1.set_xdata(xpos1)
            pendulumArm1.set_ydata(ypos1)

            pendulumArm2.set_xdata(xpos2)
            pendulumArm2.set_ydata(ypos2)

            # Update time
            time_text.set_text(f"Time: {self.time[i]:2.2f}")
            return cart, pendulumArm1, pendulumArm2, time_text

        # TODO: Adjust framerate
        num_frames = len(self.time)
        time_interval = self.time[-1] - self.time[0]
        fps = num_frames / time_interval
        interval = 1000/fps

        anim = animation.FuncAnimation(
            fig,
            animate,
            interval=interval,
            frames=len(self.states),
            init_func=init,
            blit=True
        )
        if save:
            writergif = animation.PillowWriter(fps=fps)
            anim.save(filename, writer=writergif)
            plt.close(fig)

        if not hide:
            plt.show()
        plt.close()

if __name__ == "__main__":
    initial_state = np.array([0,0,0.01,0,0,0])
    parameters = {
            "cart_mass": 1.0,
            "pendulum_1_mass": 0.1,
            "pendulum_2_mass": 0.1,
            "pendulum_1_length": 1.0,
            "pendulum_2_length": 1.0
    }

    problem = DoublePendulumOnCart(parameters, initial_state)
    problem.print()
    state = problem.get_current_state()
    t = np.linspace(0,10,100)
    for i in t:
        results = problem.step(0.1)

    problem.animate()
