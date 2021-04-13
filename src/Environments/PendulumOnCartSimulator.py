###############################
#
#   Pendulum on a Cart
#
###############################


from .OdeProblemBase import OdeProblemBase
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.lines as lines

class PendulumOnCartSimulator(OdeProblemBase):
    """
    Pendulum on a Cart problem.
    """
    g = 9.81

    def __init__(self, parameters, initial_state):
        """
        """
        super().__init__(initial_state)

        self.cart_mass = parameters["cart_mass"]
        self.pendulum_mass = parameters["pendulum_mass"]
        self.pendulum_length = parameters["pendulum_length"]

    def print(self):
        """
        Prints the model information.

        TODO: Complete summary.
        """
        print(f"Cart mass: {self.cart_mass}")
        print(f"Pendulum mass: {self.pendulum_mass}")
        print(f"Pendulum length: {self.pendulum_length}")
        print(f"Initial conditions: {self.initial_state} \n"
            "\t[\n"
            f"\t\tCart x position: {self.initial_state[0]}\n"
            f"\t\tCart velocity: {self.initial_state[1]}\n"
            f"\t\tPendulum angle: {self.initial_state[2]}\n"
            f"\t\tPendulum angular velocity: {self.initial_state[3]}\n"
            "\t]")

    def animate(self, save=False, filename=None, title="Pendulum on a Cart", hide=False):
        """
        TODO: Complete summary.
        """
        # TODO: Add error handling in case someone invokes animate before sovle!!
        # Create animation plot.
        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            aspect = 'equal',
            xlim = (-5, 5),
            ylim = (-2, 2),
            title = title)

        ax.grid()
        origin = [0.0, 0.0]

        # Create the cart.
        cart_borders = [0.0 - 0.25, 0.0 - 0.25]
        cart = patches.Rectangle(
            cart_borders,
            0.5,
            0.5,
            facecolor='none',
            edgecolor='k')

        # Pendulum arm
        theta0 = self.initial_state[2]
        initial_pendulum_bob_location = [
            self.pendulum_length * np.sin(theta0),
            self.pendulum_length * np.cos(theta0)]

        pendulumArm = lines.Line2D(
            origin,
            initial_pendulum_bob_location,
            color='r',
            marker='o')

        # Time:
        time_text = ax.text(-2., 1.6,'', fontsize=15)
        def init():
            ax.add_patch(cart)
            ax.add_line(pendulumArm)
            time_text.set_text('Time 0.0')
            return cart, pendulumArm, time_text

        def animate(i):
            cart_xpos, cart_vel, theta, theta_dot = self.states[i]

            # Cart
            cart_coordinate = [cart_xpos - 0.25, -0.25]
            cart.set_xy(cart_coordinate)

            # Pendulum
            x_pendulum_bob = cart_xpos + self.pendulum_length*np.sin(theta) # important! bob position is relative to cart xpos
            y_pendulum_bob = self.pendulum_length*np.cos(theta)
            xpos = [cart_xpos, x_pendulum_bob]
            ypos = [0.0, y_pendulum_bob]

            pendulumArm.set_xdata(xpos)
            pendulumArm.set_ydata(ypos)

            # Update time
            time_text.set_text(f"Time: {self.time[i]:2.2f}")
            return cart, pendulumArm, time_text

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
            writergif = animation.PillowWriter(fps=30)
            anim.save(filename, writer=writergif)

        if not hide:
            plt.show()

    def _rhs(self, state, t, u):
        """
        TODO: Complete summary.
        """
        force = u
        xpos, xpos_dot, theta, theta_dot = state

        # Equations from: https://coneural.org/florian/papers/05_cart_pole.pdf
        sub_denominator = self.pendulum_mass + self.cart_mass
        tmp1 = -force - self.pendulum_mass*self.pendulum_length*theta_dot**2*np.sin(theta) / (sub_denominator)
        denominator = self.pendulum_length * (4/3 - (self.pendulum_mass * np.cos(theta)**2) / sub_denominator)

        theta_dot_dot = (self.g * np.sin(theta) + np.cos(theta) * tmp1) / denominator

        tmp2 = theta_dot**2 * np.sin(theta) - theta_dot_dot * np.cos(theta)
        xpos_dot_dot = (force + self.pendulum_mass * self.pendulum_length * (tmp2)) / denominator

        dydx = np.array([
                xpos_dot,
                xpos_dot_dot,
                theta_dot,
                theta_dot_dot
               ])
        return dydx

# Use the class
if __name__ == "__main__":
    """
    Simulates a pendulum on a cart problem.
    """
    initial_state = np.array([0.0, 0.0, 0.05, 0.0])
    parameters = {
            "cart_mass": 1.0,
            "pendulum_mass": 0.1,
            "pendulum_length" : 1.0,
        }

    problem = PendulumOnCart(parameters, initial_state)
    problem.print()
    state = problem.get_current_state()

    # # Time line
    t = np.linspace(0, 10, 100)

    for i in t:
        results = problem.step(0.1)
    # problem.animate()

    # problem.reset()

    # problem.solve(t, 0.1)
    problem.animate()
