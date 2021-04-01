# A single pendulum on a cart.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.integrate import odeint
import matplotlib.lines as lines


#   * Paramater values
#       [
#           cart_mass,
#           pendulum_length,
#           pendulum_bob_mass,
#           gravity
#       ]
#   * State representation
#       x0 = [
#              x,           # cart positon
#              x_dot,       # cart velocity
#              theta,       # pendulum angle
#              theta_dot    # pendulum angular velocity
#            ]
#
#   * Initial conditions
#
#   * Right-hand-side function
#       x0_dot = [
#                  x_dot,           # cart velocity
#                  x_dot_dot,       # cart acceleration
#                  theta_dot,       # angular velocity
#                  theta_dot_dot    # angular acceleration
#                ]
#       def rhs(x, t):
#          """
#          """
#          u = control_force(x, t)
#          x_dot_vector = Ax + Bu
#          return x_dot_vector
#
#   * Step function(h (timestep))
#       Takes the current state and advances one step in time.
#
#   * reset
#       reset to initial state
#
#   * Solve()
#       solves the differential equations
#
#   * Animate(save_file=None)
#       Animates the problem state over time.
#
#   * get_states
#       Get the current state of the problem.
#
#   * get_all_states
#       * get all states of the pendulum


class PendulumOnCart():
    """
    Single pendulum on a cart model.
    """
    g = 9.81

    def __init__(self, initial_state, parameters, controller=None):
        """
        Constructs a pendulum on a cart model with specified parameters.

        TODO: Fix function summary with detailed argument description.
        """
        self.cart_mass = parameters["cart_mass"]
        self.pendulum_mass = parameters["pendulum_mass"]
        self.pendulum_length = parameters["pendulum_length"]

        self.initial_state = initial_state

        if controller == None:
            self.external_force = lambda x, t : 0
        else:
            self.external_force = controller
    
    def rhs(self, x, t):
        """
        """
        # force = self.external_force(x, t)
        force = 0

        xpos, xpos_dot, theta, theta_dot = x

        # Equations from: https://coneural.org/florian/papers/05_cart_pole.pdf
        sub_denominator = self.pendulum_mass + self.cart_mass
        tmp1 = -force - self.pendulum_mass*self.pendulum_length*theta_dot**2*np.sin(theta) / (sub_denominator)
        denominator = self.pendulum_length * (4/3 - (self.pendulum_mass * np.cos(theta)**2) / sub_denominator)

        theta_dot_dot = (self.g * np.sin(theta) + np.cos(theta) * tmp1) / denominator

        tmp2 = theta_dot**2 * np.sin(theta) - theta_dot_dot * np.cos(theta)
        xpos_dot_dot = (force + self.pendulum_mass * self.pendulum_length * (tmp2)) / denominator

        dydx = [
                xpos_dot,
                xpos_dot_dot,
                theta_dot,
                theta_dot_dot
               ]
        return dydx
    
    def get_initial_state(self):
        """
        Returns the initial state of the problem.
        """
        return self.initial_state
    
    def print(self):
        """
        Prints the model information.
        """
        print(f"Cart mass: {self.cart_mass}")
        print(f"Pendulum mass: {self.pendulum_mass}")
        print(f"Pendulum length: {self.pendulum_length}")
        print(f"Initial conditions: {self.initial_state}")

    def solve(self, t):
        """
        Solves the problems for time t.

        TODO: Add description.
        """
        # Solve the equations
        self.sol = odeint(self.rhs, self.initial_state, t)
        self.time = t

    def animate(self):
        """
        """
        # TODO: Add error handling in case someone invokes animate before sovle!!
        # Create animation plot.
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect = 'equal', xlim = (-2, 2), ylim = (-2, 2), title = "Pendulum on a Cart")
        ax.grid()

        origin = [0.0, 0.0]

        # Create the cart.
        cart_borders = [0.0 - 0.25, 0.0 - 0.25]
        cart = patches.Rectangle(cart_borders, 0.5, 0.5, facecolor='none', edgecolor='k')

        # Pendulum arm
        initial_pendulum_bob_location = [self.pendulum_length * np.sin(self.initial_state[2]), self.pendulum_length * np.cos(self.initial_state[2])]
        pendulumArm = lines.Line2D(origin, initial_pendulum_bob_location, color='r', marker='o') 

        # Time:
        time_text = ax.text(-2., 1.6,'', fontsize=15)
        def init():
            ax.add_patch(cart)
            ax.add_line(pendulumArm)
            time_text.set_text('Time 0.0')
            return cart, pendulumArm, time_text

        def animate(i):
            # Cart
            cart_xpos = self.sol[i, 0]
            cart_coordinate = [cart_xpos - 0.25, -0.25]
            cart.set_xy(cart_coordinate)

            # Pendulum
            theta = self.sol[i, 2]
            x_pendulum_bob = cart_xpos + self.pendulum_length*np.sin(theta) # important! bob position is relative to cart xpos
            y_pendulum_bob = self.pendulum_length*np.cos(theta)
            xpos = [cart_xpos, x_pendulum_bob]
            ypos = [0.0, y_pendulum_bob]

            pendulumArm.set_xdata(xpos)
            pendulumArm.set_ydata(ypos)

            # Update time
            time_text.set_text(f"Time: {self.time[i]:2.2f}")
            return cart, pendulumArm, time_text

        anim = animation.FuncAnimation(
            fig,
            animate,
            interval=0.1,                    # TODO: Fix!
            frames=len(self.sol),
            init_func=init)
        plt.show()


# Use the class

initial_state = [0.0, 0.0, 0.1, 0.0]
parameters = {
        "cart_mass": 1.0,
        "pendulum_mass": 1.0,
        "pendulum_length" : 1.0,
    }
problem = PendulumOnCart(initial_state, parameters)
problem.print()

# Time line
t = np.linspace(0, 10, 200)

problem.solve(t)
problem.animate()
