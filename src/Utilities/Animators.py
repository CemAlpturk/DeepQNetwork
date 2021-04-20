import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.lines as lines

from Environments import DoublePendulumOnCartSimulator
from Environments import PendulumOnCartSimulator

def animate_multiple_single_pendulums(
    pendulums,
    filename=None,
    hide=False):
    """
    Animates multiple single pendulums in the same figure.
    """
    colors = ['r', 'b', 'g', 'y']

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        aspect='equal',
        xlim = (-5, 5),
        ylim = (-2.5, 2.5),
        title = "Multiple Single Pendulum on a Cart"
    )

    ax.grid()
    origin = [0.0, 0.0]

    # Setting up carts:
    carts = [None] * len(pendulums)
    for index in range(len(pendulums)):
        cart_borders = [0.0 - 0.25, 0.0 - 0.25]
        cart = patches.Rectangle(
            cart_borders,
            0.5,
            0.5,
            facecolor='none',
            edgecolor='k')

        carts[index] = cart

    # Setting up pendulum 1 and 2 arms
    pendulum_arms = [None] * len(pendulums)

    for index, pendulum in enumerate(pendulums):
        # Set the pendulum color:
        color = colors[index % len(colors)]

        # Grab the angle for the inner pendulum
        theta = pendulum.initial_state[2]

        # Bob location for the inner pendulum
        initial_pendulum_bob1_location = [
            pendulum.pendulum_length * np.sin(theta),
            pendulum.pendulum_length * np.cos(theta)
        ]

        # Create line for pendulum
        pendulumArm = lines.Line2D(
            origin,
            initial_pendulum_bob1_location,
            color=color,
            marker='o'
        )
        pendulum_arms[index] = pendulumArm

    # Setup time animation
    time_text = ax.text(-4, 1.6, '', fontsize=12)

    # Setup init function for animator.
    def init():
        # Add carts
        for index, cart in enumerate(carts):
            ax.add_patch(cart)

        # Add pendulum arms:
        for index, arm_inner in enumerate(pendulum_arms):
            ax.add_line(arm_inner)

        # Set initial time.
        time_text.set_text('Time 0.0')
        ret_val = carts + pendulum_arms + [time_text]
        return tuple(ret_val)

    # Setup the animator function
    def animate(i):
        for index, pendulum in enumerate(pendulums):
            # Get the states
            cart_xpos, x_dot, q1, q1_dot = pendulum.states[i]

            # Update Cart
            cart_coordinate = [cart_xpos - 0.25, -0.25]
            carts[index].set_xy(cart_coordinate)

            # Update pendulum
            x_pendulum_bob = cart_xpos + pendulum.pendulum_length*np.sin(q1)
            y_pendulum_bob = pendulum.pendulum_length*np.cos(q1)
            xpos = [cart_xpos, x_pendulum_bob]
            ypos = [0.0, y_pendulum_bob]

            pendulum_arms[index].set_xdata(xpos)
            pendulum_arms[index].set_ydata(ypos)
        
        # Update time
        time_text.set_text(f"Time: {pendulums[0].time[i]:2.2f}")
        ret_val = carts + pendulum_arms + [time_text]
        return tuple(ret_val)

    # TODO: Adjust framerate
    _, time_array = pendulums[0].get_states_and_time()

    num_frames = len(time_array)
    time_interval = time_array[-1] - time_array[0]
    fps = num_frames / time_interval
    interval = 1000/fps

    anim = animation.FuncAnimation(
        fig,
        animate,
        interval=interval,
        frames=len(pendulums[0].states),
        init_func=init,
        blit=True)

    if filename is not None:
        writergif = animation.PillowWriter(fps=fps)
        anim.save(filename, writer=writergif)
    
    if not hide:
        plt.show()



def animate_multiple_double_pendulums(
    pendulums,
    filename=None,
    hide=False):
    """
    Animates multiple double pendulums in the same figure.
    """
    colors = ['r', 'b', 'g', 'y']

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        aspect='equal',
        xlim = (-5, 5),
        ylim = (-2.5, 2.5),
        title = "Multiple Double Pendulum on a Cart"
    )

    ax.grid()
    origin = [0.0, 0.0]

    # Setting up carts:
    carts = [None] * len(pendulums)
    for index in range(len(pendulums)):
        cart_borders = [0.0 - 0.25, 0.0 - 0.25]
        cart = patches.Rectangle(
            cart_borders,
            0.5,
            0.5,
            facecolor='none',
            edgecolor='k')

        carts[index] = cart

    # Setting up pendulum 1 and 2 arms
    pendulum_arms = [None] * len(pendulums)
    pendulum_2_arms = [None] * len(pendulums)

    for index, pendulum in enumerate(pendulums):
        # Set the pendulum color:
        color = colors[index % len(colors)]

        # Grab the angle for the inner pendulum
        theta_01 = pendulum.initial_state[1]

        # Bob location for the inner pendulum
        initial_pendulum_bob1_location = [
            pendulum.l1 * np.sin(theta_01),
            pendulum.l1 * np.cos(theta_01)
        ]

        # Bob location for the outer pendulum
        theta_02 = pendulum.initial_state[2]
        initial_pendulum_bob2_location = [
            initial_pendulum_bob1_location[0] + pendulum.l2*np.sin(theta_02),
            initial_pendulum_bob1_location[1] + pendulum.l2*np.cos(theta_02)
        ]

        # Create line for inner pendulum
        pendulumArm = lines.Line2D(
            origin,
            initial_pendulum_bob1_location,
            color=color,
            marker='o'
        )
        pendulum_arms[index] = pendulumArm

        # Create line for outer pendulum
        pendulumArm2 = lines.Line2D(
            initial_pendulum_bob1_location,
            initial_pendulum_bob2_location,
            color=color,
            marker='o'
        )
        pendulum_2_arms[index] = pendulumArm2

    # Setup time animation
    time_text = ax.text(-4, 1.6, '', fontsize=12)

    # Setup init function for animator.
    def init():
        # Add carts
        for index, cart in enumerate(carts):
            ax.add_patch(cart)
        
        # ax.add_patch(cart)

        # Add inner pendulum arms:
        for index, arm_inner in enumerate(pendulum_arms):
            ax.add_line(arm_inner)
        
        # Add outer pendulum arms:
        for index, arm_outer in enumerate(pendulum_2_arms):
            ax.add_line(arm_outer)

        # Set initial time.
        time_text.set_text('Time 0.0')
        ret_val = carts + pendulum_arms + pendulum_2_arms + [time_text]
        return tuple(ret_val)

    # Setup the animator function
    def animate(i):
        for index, pendulum in enumerate(pendulums):
            # Get the states
            cart_xpos, q1, q2, x_dot, q1_dot, q2_dot = pendulum.states[i]

            # Update Cart
            cart_coordinate = [cart_xpos - 0.25, -0.25]
            carts[index].set_xy(cart_coordinate)

            # Update inner pendulum
            x_pendulum_bob1 = cart_xpos + pendulum.l1*np.sin(q1)
            y_pendulum_bob1 = pendulum.l1*np.cos(q1)
            xpos1 = [cart_xpos, x_pendulum_bob1]
            ypos1 = [0.0, y_pendulum_bob1]

            pendulum_arms[index].set_xdata(xpos1)
            pendulum_arms[index].set_ydata(ypos1)

            # Update outer pendulum
            x_pendulum_bob2 = x_pendulum_bob1 + pendulum.l2*np.sin(q2)
            y_pendulum_bob2 = y_pendulum_bob1 + pendulum.l2*np.cos(q2)
            xpos2 = [x_pendulum_bob1, x_pendulum_bob2]
            ypos2 = [y_pendulum_bob1, y_pendulum_bob2]

            pendulum_2_arms[index].set_xdata(xpos2)
            pendulum_2_arms[index].set_ydata(ypos2)
        
        # Update time
        time_text.set_text(f"Time: {pendulums[0].time[i]:2.2f}")
        ret_val = carts + pendulum_arms + pendulum_2_arms + [time_text]
        return tuple(ret_val)

    # TODO: Adjust framerate
    _, time_array = pendulums[0].get_states_and_time()

    num_frames = len(time_array)
    time_interval = time_array[-1] - time_array[0]
    fps = num_frames / time_interval
    interval = 1000/fps

    anim = animation.FuncAnimation(
        fig,
        animate,
        interval=interval,
        frames=len(pendulums[0].states),
        init_func=init,
        blit=True)

    if filename is not None:
        writergif = animation.PillowWriter(fps=fps)
        anim.save(filename, writer=writergif)
    
    if not hide:
        plt.show()
