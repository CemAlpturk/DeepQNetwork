import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.lines as lines

class SinglePendulumAnimator:
    """
    Animates a single pendulum.
    """

    _default_plot_settings = {
        "show_grid" : True,
        "x_max" : 5,
        "y_max" : 2,

        "cart_width" : 0.5,
        "cart_height" : 0.5,
        "cart_border_color" : "k",

        "pendulum_color" : "r",

        "force_bar_show" : False,
        "force_bar_color" : "r",
        "force_bar_alpha" : 0.5,
        "force_action_space" : [],

        "show_termination_boundary" : False,
        "termination_angle" : 10 * np.pi/180,
        "termination_boundary_color" : 'r',
        "termination_boundary_alpha" : 0.5,

        "time_font_size" : 16
    }

    @staticmethod
    def animate_from_csv(
            data_filename : str,
            pendulum_settings : dict,
            title = "Pendulum on Cart",
            output_filename = None,
            save = False,
            hide = False,
            plot_settings = None):
        """
        Reads data from CSV file and animate the single pendulum
        with provided settings.

        TODO: fill out summary
        """
        # TODO: Add input argument checking
        #       Require that the pendulum settings contain 'Pendulum_Length'
        #       Require that max_angle is provided if animate_Termination_boundary=True (Maybe it's enough to check if it's true/false)
        #       Require that an action space is provided if 'animate_force = True'
        #           (unless the current way of guessing the max is to be used
        #               (problem when no force is applied throughout the entire simulation))

        # TODO: Validate pendulum settings.
        plt_settings = SinglePendulumAnimator._get_plot_settings(plot_settings)

        # Read data.
        data = pd.read_csv(
            data_filename,
            usecols=['States', 'Force', 'Reward', 'Time'])

        # Extract data to separate components.
        forces = data['Force'][:].to_numpy(dtype=object)
        raw_states = data['States'][:].to_numpy(dtype=object)
        times = data['Time'][:].to_numpy()

        # The CSV file reads the state inputs as a string.
        # Convert the string to numpy array
        states = [np.array(list(map(float, x[2:-2].split()))) for x in raw_states[:]]

        # Animate the single pendulum.
        SinglePendulumAnimator._animate(
            states,
            times,
            pendulum_settings,
            forces,
            save,
            output_filename,
            title,
            hide,
            plt_settings)

    @staticmethod
    def animate_simulation(
            simulation,
            pendulum_settings,
            plot_settings=None,
            save=False,
            output_filename=None,
            title="Pendulum on Cart",
            hide=False):
        """
        Animates a simulation.

        TODO: Fix summary.
        """
        states = simulation.states
        times = simulation.time
        forces = simulation.external_forces

        plt_settings = SinglePendulumAnimator._get_plot_settings(plot_settings)

        SinglePendulumAnimator._animate(
            states,
            times,
            pendulum_settings,
            forces,
            save,
            output_filename,
            title,
            hide,
            plt_settings)

    @staticmethod
    def _get_plot_settings(custom_plot_settings : dict):
        """
        Gets the plot settings to use for the plot.
        Custom plot settings are user provided settings that override the default values.
        A user does not need to provide all settings, the ones that are provided override
        existing settings. Default settings are used for settings that are not provided by
        the user.
        """
        if custom_plot_settings == None:
            return SinglePendulumAnimator._default_plot_settings

        plot_settings = SinglePendulumAnimator._default_plot_settings
        for key in custom_plot_settings:
            plot_settings[key] = custom_plot_settings[key]

        return plot_settings

    def _animate(
        states,
        times,
        pendulum_settings,
        external_forces,
        save,
        filename,
        title,
        hide,
        plot_settings):
        """
        TODO: Complete summary.
        """
        # TODO: Add error handling in case someone invokes animate before sovle!!
        # Create animation plot.
        xlim_max = 5
        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            aspect = 'equal',
            xlim = (-plot_settings["x_max"], plot_settings["x_max"]),
            ylim = (-plot_settings["y_max"], plot_settings["y_max"]),
            title = title)

        if plot_settings['show_grid']:
            ax.grid()

        # Grab the initial state.
        initial_state = states[0]
        initial_cart_x_pos = initial_state[0]
        initial_cart_y_pos = 0                  # Cart does not move in y-axis

        # Create the cart.
        cart_width_half = plot_settings["cart_width"] / 2.0
        cart_height_half = plot_settings["cart_height"] / 2.0
        cart_borders = [
            initial_cart_x_pos - cart_width_half,
            initial_cart_y_pos - cart_height_half]

        cart = patches.Rectangle(
            cart_borders,
            plot_settings["cart_width"] ,
            plot_settings["cart_height"],
            facecolor='none',
            edgecolor=plot_settings['cart_border_color'])

        # Create animation components for applied force.
        if plot_settings["force_bar_show"]:
            force_bar_border = [0.0, -plot_settings["y_max"]]
            force_bar = patches.Rectangle(
                force_bar_border,
                0.0,
                1.0,
                facecolor=plot_settings["force_bar_color"],
                alpha=plot_settings["force_bar_alpha"])
            force_divider = patches.Rectangle(
                [0.0, -plot_settings["y_max"]],
                0.1,
                1.0,
                facecolor='k')

            # Scales the applied force so it fits in the plot.
            # TODO: Maybe this should break if the action space isn't provided?
            if len(plot_settings["force_action_space"]) != 0:
                force_scaler = np.max(np.abs(plot_settings["force_action_space"]))
            else:
                force_scaler = 1.0

        # Pendulum arm
        initial_state = states[0]
        theta0 = initial_state[2]
        initial_pendulum_bob_location = [
            pendulum_settings["pendulum_length"] * np.sin(theta0),
            pendulum_settings["pendulum_length"] * np.cos(theta0)]

        pendulumArm = lines.Line2D(
            [initial_state[0], 0.0],
            initial_pendulum_bob_location,
            color=plot_settings["pendulum_color"],
            marker='o')

        # Add termination boundary.
        if plot_settings["show_termination_boundary"]:
            x_boundary = np.sin(plot_settings["termination_angle"])
            left_boundary = lines.Line2D(
                [-x_boundary, initial_cart_x_pos],
                [1.5, initial_cart_y_pos],
                color=plot_settings["termination_boundary_color"],
                marker='',
                ls="--",
                alpha=plot_settings["termination_boundary_alpha"])

            right_boundary = lines.Line2D(
                [initial_cart_x_pos, x_boundary],
                [initial_cart_y_pos, 1.5],
                color=plot_settings["termination_boundary_color"],
                marker="",
                ls="--",
            alpha=plot_settings["termination_boundary_alpha"])

        # Time:
        time_text = ax.text(
            -plot_settings["x_max"],
            plot_settings["y_max"]-1,
            '',
            fontsize=plot_settings["time_font_size"])

        def init():
            ax.add_patch(cart)
            ax.add_line(pendulumArm)
            time_text.set_text('Time 0.0')

            if plot_settings["show_termination_boundary"]:
                ax.add_line(left_boundary)
                ax.add_line(right_boundary)

            # Only add the force animation if set to: True.
            if plot_settings["force_bar_show"]:
                ax.add_patch(force_bar)
                ax.add_patch(force_divider)

            if plot_settings["force_bar_show"] and plot_settings["show_termination_boundary"]:
                return force_divider, force_bar, left_boundary, right_boundary, cart, pendulumArm, time_text
            elif plot_settings["force_bar_show"]:
                return force_divider, force_bar, cart, pendulumArm, time_text
            elif plot_settings["show_termination_boundary"]:
                return left_boundary, right_boundary, cart, pendulumArm, time_text

            return cart, pendulumArm, time_text

        def animate(i):
            cart_xpos, cart_vel, theta, theta_dot = states[i]

            # Cart
            cart_coordinate = [cart_xpos - cart_width_half, -cart_height_half]
            cart.set_xy(cart_coordinate)

            # Pendulum
            x_pendulum_bob = cart_xpos + pendulum_settings["pendulum_length"]*np.sin(theta) # important! bob position is relative to cart xpos
            y_pendulum_bob = pendulum_settings["pendulum_length"]*np.cos(theta)
            xpos = [cart_xpos, x_pendulum_bob]
            ypos = [0.0, y_pendulum_bob]                                                    # Cart y-pos is always zero!

            pendulumArm.set_xdata(xpos)
            pendulumArm.set_ydata(ypos)

            # Update time
            time_text.set_text(f"Time: {times[i]:2.2f}")

            # Update termination boundary (the y-data doesn't change).
            if plot_settings["show_termination_boundary"]:
                left_boundary.set_xdata([-x_boundary + cart_xpos, cart_xpos])
                right_boundary.set_xdata([cart_xpos, x_boundary + cart_xpos])

            # Only update force animation if set to: True.
            if plot_settings["force_bar_show"]:
                # Update the force_bar.
                scaled_force = xlim_max * external_forces[i] / force_scaler
                force_bar.set_width(scaled_force)

                # Set the applied force amount to the label.
                ax.set_xlabel(f'Applied force: {external_forces[i]}')

            if plot_settings["force_bar_show"] and plot_settings["show_termination_boundary"]:
                return force_divider, force_bar, left_boundary, right_boundary, cart, pendulumArm, time_text
            elif plot_settings["force_bar_show"]:
                return force_divider, force_bar, cart, pendulumArm, time_text
            elif plot_settings["show_termination_boundary"]:
                return left_boundary, right_boundary, cart, pendulumArm, time_text

            return cart, pendulumArm, time_text

        times, states, external_forces, fps, interval = SinglePendulumAnimator._trim_data(times, states, external_forces)

        anim = animation.FuncAnimation(
            fig,
            animate,
            # interval=1.0,
            frames=len(states),
            init_func=init,
            blit=True)

        if save:
            writergif = animation.PillowWriter(fps=fps)
            anim.save(filename, writer=writergif)
            plt.close(fig)

        if not hide:
            plt.show()

    @staticmethod
    def _trim_data(times, states, forces):
        """
        Trims the data to match FPS (60).
        """
        num_frames = len(times)
        time_interval = times[-1] - times[0]
        fps = num_frames / time_interval

        while fps > 60:
            times = times[1::2]
            states = states[1::2]
            forces = forces[1::2]

            num_frames = len(times)
            time_interval = times[-1] - times[0]
            fps = num_frames / time_interval

        interval = 1000/fps
        return (times, states, forces, fps, interval)

    @staticmethod
    def _create_cart(
            initial_cart_x_pos,
            initial_cart_y_pos,
            plot_settings) -> patches.Rectangle:
        """
        Creates a cart object for animation.
        """
        # Cart cannot move in y-axis.
        initial_cart_y_pos = 0

        cart_width_half = plot_settings["cart_width"] / 2.0
        cart_height_half = plot_settings["cart_height"] / 2.0
        cart_borders = [
            initial_cart_x_pos - cart_width_half,
            initial_cart_y_pos - cart_height_half]

        cart = patches.Rectangle(
            cart_borders,
            plot_settings["cart_width"] ,
            plot_settings["cart_height"],
            facecolor='none',
            edgecolor=plot_settings['cart_border_color'])

        return cart


class DoublePendulumAnimator:
    """
    Anmiates a inverted double pendulum on a cart.
    """

    _default_plot_settings = {
        "show_grid" : True,
        "x_max" : 5,
        "y_max" : 3,

        "cart_width" : 0.5,
        "cart_height" : 0.5,
        "cart_border_color" : 'k',

        "pendulum_inner_color" : 'r',
        "pendulum_outer_color" : 'r',

        "force_bar_show" : False,
        "force_bar_color" : "r",
        "force_bar_alpha" : 0.5,
        "force_action_space" : [],

        "show_termination_boundary" : False,
        "termination_boundary_color" : 'r',
        "termination_boundary_alpha" : 0.5,
        "termination_angle_inner_pendulum" : 5*np.pi/180,
        "termination_angle_outer_pendulum" : 5*np.pi/180,

        "time_font_size" : 16,
        "time_x_coordinate" : -4,
        "time_y_coordinate" : 1.6
    }

    @staticmethod
    def animate_from_csv(
            filename,
            pendulum_settings : dict,
            title = "Pendulum on Cart",
            output_filename = None,
            save = False,
            hide = False,
            plot_settings = None):
        """
        Reads a CSV file and animates the double pendulum with provided settings.
        """
        data = pd.read_csv(
            filename,
            usecols=['States', 'Force', 'Reward', 'Time'])

        forces = data['Force'][:].to_numpy(dtype=object)
        raw_states = data['States'][:].to_numpy(dtype=object)
        times = data['Time'][:].to_numpy()

        # The CSV file reads the input as a string.
        # Convert the string to numpy array
        states = [np.array(list(map(float, x[2:-2].split()))) for x in raw_states[:]]

        plt_settings = DoublePendulumAnimator._get_plot_settings(plot_settings)


        DoublePendulumAnimator._animate(
            states,
            times,
            forces,
            pendulum_settings,
            plt_settings,
            save,
            output_filename,
            title,
            hide)

    def animate_simulation(
            simulation,
            pendulum_settings,
            plot_settings=None,
            save=False,
            output_filename=None,
            title="Double Pendulum on Cart",
            hide=False):
        """
        Animates a simulation.

        TODO: Fix summary.
        """
        states = simulation.states
        times = simulation.time
        forces = simulation.external_forces

        plt_settings = DoublePendulumAnimator._get_plot_settings(plot_settings)

        DoublePendulumAnimator._animate(
            states,
            times,
            forces,
            pendulum_settings,
            plt_settings,
            save,
            output_filename,
            title,
            hide)

    @staticmethod
    def _get_plot_settings(custom_plot_settings : dict):
        """
        Gets the plot settings to use for the plot.
        Custom plot settings are user provided settings that override the default values.
        A user does not need to provide all settings, the ones that are provided override
        existing settings. Default settings are used for settings that are not provided by
        the user.
        """
        if custom_plot_settings == None:
            return DoublePendulumAnimator._default_plot_settings
        plot_settings = DoublePendulumAnimator._default_plot_settings
        for key in custom_plot_settings:
            plot_settings[key] = custom_plot_settings[key]

        return plot_settings

    def _animate(
        states,
        times,
        external_forces,
        pendulum_settings,
        plot_settings,
        save,
        filename,
        title,
        hide):
        """
        TODO
        """
        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            aspect='equal',
            xlim = (-plot_settings["x_max"], plot_settings["x_max"]),
            ylim = (-plot_settings["y_max"],plot_settings["y_max"]),
            title = title
        )

        if plot_settings['show_grid']:
            ax.grid()

        # Grab the initial state.
        initial_state = states[0]
        initial_cart_xpos = initial_state[0]
        initial_cart_ypos = 0.0                 # Cart cannot move in y-axis.

        # Create the cart
        cart_width_half = plot_settings["cart_width"] / 2.0
        cart_height_half = plot_settings["cart_height"] / 2.0

        cart_borders = [initial_cart_xpos - cart_width_half, initial_cart_ypos - cart_height_half]
        cart = patches.Rectangle(
            cart_borders,
            plot_settings["cart_width"],
            plot_settings["cart_height"],
            facecolor='none',
            edgecolor=plot_settings['cart_border_color']
        )

        # Pendulum arm
        theta01 = initial_state[1]
        initial_pendulum_bob1_location = [
            pendulum_settings["inner_pendulum_length"] * np.sin(theta01),
            pendulum_settings["inner_pendulum_length"] * np.cos(theta01)
        ]

        theta02 = initial_state[2]
        initial_pendulum_bob2_location = [
            initial_pendulum_bob1_location[0] + pendulum_settings["outer_pendulum_length"]*np.sin(theta02),
            initial_pendulum_bob1_location[1] + pendulum_settings["outer_pendulum_length"]*np.cos(theta02)
        ]

        pendulumArm1 = lines.Line2D(
            [initial_cart_xpos, initial_cart_ypos],
            initial_pendulum_bob1_location,
            color=plot_settings['pendulum_inner_color'],
            marker='o'
        )

        pendulumArm2 = lines.Line2D(
            initial_pendulum_bob1_location,
            initial_pendulum_bob2_location,
            color=plot_settings['pendulum_outer_color'],
            marker='o'
        )

        # Add boundary (needs to be relative to the cart.)
        if plot_settings["show_termination_boundary"]:
            x_boundary_inner_pendulum = np.sin(plot_settings["termination_angle_inner_pendulum"])
            left_boundary_inner_pendulum = lines.Line2D(
                [-x_boundary_inner_pendulum, initial_cart_xpos],
                [pendulum_settings["inner_pendulum_length"] + 0.5, initial_cart_ypos],
                color=plot_settings["termination_boundary_color"],
                marker='',
                ls="--",
                alpha=plot_settings["termination_boundary_alpha"])

            right_boundary_inner_pendulum = lines.Line2D(
                [initial_cart_xpos, x_boundary_inner_pendulum],
                [initial_cart_ypos, pendulum_settings["inner_pendulum_length"] + 0.5],
                color=plot_settings["termination_boundary_color"],
                marker="",
                ls="--",
                alpha=plot_settings["termination_boundary_alpha"])


            x_boundary_outer_pendulum = np.sin(plot_settings["termination_angle_outer_pendulum"])
            left_boundary_outer_pendulum = lines.Line2D(
                [-x_boundary_outer_pendulum, initial_pendulum_bob2_location[0]],
                [initial_pendulum_bob2_location[1] + 0.5, initial_pendulum_bob1_location[1]],
                color=plot_settings["termination_boundary_color"],
                marker='',
                ls="--",
                alpha=plot_settings["termination_boundary_alpha"])

            right_boundary_outer_pendulum = lines.Line2D(
                [initial_pendulum_bob1_location[0], initial_pendulum_bob1_location[0] + x_boundary_outer_pendulum],
                [initial_pendulum_bob1_location[1], initial_pendulum_bob2_location[1] + 0.5],
                color=plot_settings["termination_boundary_color"],
                marker="",
                ls="--",
                alpha=plot_settings["termination_boundary_alpha"])

        # Create animation components for applied force.
        if plot_settings["force_bar_show"]:
            force_bar_border = [0.0, -plot_settings["y_max"]]
            force_bar = patches.Rectangle(
                force_bar_border,
                0.0,
                1.0,
                facecolor=plot_settings["force_bar_color"],
                alpha=plot_settings["force_bar_alpha"])
            force_divider = patches.Rectangle(
                [0.0, -plot_settings["y_max"]],
                0.1,
                1.0,
                facecolor='k')

            # Scales the applied force so it fits in the plot.
            # TODO: Maybe this should break if the action space isn't provided?
            if len(plot_settings["force_action_space"]) != 0:
                force_scaler = np.max(np.abs(plot_settings["force_action_space"]))
            else:
                force_scaler = 1.0

        # Time
        time_text = ax.text(
            -plot_settings["x_max"],
            plot_settings["y_max"]-1,
            '',
            fontsize=plot_settings["time_font_size"])

        def init():
            ax.add_patch(cart)
            ax.add_line(pendulumArm1)
            ax.add_line(pendulumArm2)

            if plot_settings["show_termination_boundary"]:
                ax.add_line(left_boundary_inner_pendulum)
                ax.add_line(right_boundary_inner_pendulum)

                ax.add_line(left_boundary_outer_pendulum)
                ax.add_line(right_boundary_outer_pendulum)

                time_text.set_text('Time 0.0')

            # Only add the force animation if set to: True.
            if plot_settings["force_bar_show"]:
                ax.add_patch(force_bar)
                ax.add_patch(force_divider)

            if plot_settings["force_bar_show"] and plot_settings["show_termination_boundary"]:
                return force_divider, force_bar, left_boundary_inner_pendulum, right_boundary_inner_pendulum, left_boundary_outer_pendulum, right_boundary_outer_pendulum, cart, pendulumArm1, pendulumArm2, time_text
            elif plot_settings["force_bar_show"]:
                return force_divider, force_bar, cart, pendulumArm1, pendulumArm2, time_text
            elif plot_settings["show_termination_boundary"]:
                return left_boundary_inner_pendulum, right_boundary_inner_pendulum, left_boundary_outer_pendulum, right_boundary_outer_pendulum, cart, pendulumArm1, pendulumArm2, time_text

            return cart, pendulumArm1, pendulumArm2, time_text

        def animate(i):
            cart_xpos, q1, q2, x_dot, q1_dot, q2_dot = states[i]

            # Cart
            cart_coordinate = [cart_xpos - 0.25, -0.25]
            cart.set_xy(cart_coordinate)

            # Pendulum
            x_pendulum_bob1 = cart_xpos + pendulum_settings["inner_pendulum_length"]*np.sin(q1)
            y_pendulum_bob1 = pendulum_settings["inner_pendulum_length"]*np.cos(q1)
            xpos1 = [cart_xpos, x_pendulum_bob1]
            ypos1 = [0.0, y_pendulum_bob1]

            x_pendulum_bob2 = x_pendulum_bob1 + pendulum_settings["outer_pendulum_length"]*np.sin(q2)
            y_pendulum_bob2 = y_pendulum_bob1 + pendulum_settings["outer_pendulum_length"]*np.cos(q2)
            xpos2 = [x_pendulum_bob1, x_pendulum_bob2]
            ypos2 = [y_pendulum_bob1, y_pendulum_bob2]

            pendulumArm1.set_xdata(xpos1)
            pendulumArm1.set_ydata(ypos1)

            pendulumArm2.set_xdata(xpos2)
            pendulumArm2.set_ydata(ypos2)

            # Update time
            time_text.set_text(f"Time: {times[i]:2.2f}")

            if plot_settings["show_termination_boundary"]:
                # Update termination boundary
                left_boundary_inner_pendulum.set_xdata([-x_boundary_inner_pendulum + cart_xpos, cart_xpos])
                right_boundary_inner_pendulum.set_xdata([cart_xpos, x_boundary_inner_pendulum + cart_xpos])

                # Left Outer
                left_boundary_outer_pendulum.set_xdata([-x_boundary_outer_pendulum + x_pendulum_bob1, x_pendulum_bob1])
                left_boundary_outer_pendulum.set_ydata([y_pendulum_bob2 + 0.5, y_pendulum_bob1])

                # Right Outer
                right_boundary_outer_pendulum.set_xdata([x_pendulum_bob1, x_boundary_outer_pendulum + x_pendulum_bob1])
                right_boundary_outer_pendulum.set_ydata([y_pendulum_bob1, y_pendulum_bob2 + 0.5])

            # Only update force animation if set to: True.
            if plot_settings["force_bar_show"]:
                # Update the force_bar.
                # Scale so that max force_bar is mapped to 'xlim_max' (for the plot)
                scaled_force = plot_settings["x_max"] * external_forces[i] / force_scaler
                force_bar.set_width(scaled_force)

                # Set the applied force amount to the label.
                ax.set_xlabel(f'Applied force: {external_forces[i]}')

            if plot_settings["force_bar_show"] and plot_settings["show_termination_boundary"]:
                return force_divider, force_bar, left_boundary_inner_pendulum, right_boundary_inner_pendulum, left_boundary_outer_pendulum, right_boundary_outer_pendulum, cart, pendulumArm1, pendulumArm2, time_text
            elif plot_settings["force_bar_show"]:
                return force_divider, force_bar, cart, pendulumArm1, pendulumArm2, time_text
            elif plot_settings["show_termination_boundary"]:
                return left_boundary_inner_pendulum, right_boundary_inner_pendulum, left_boundary_outer_pendulum, right_boundary_outer_pendulum, cart, pendulumArm1, pendulumArm2, time_text

            return cart, pendulumArm1, pendulumArm2, time_text

        times, states, external_forces, fps, interval = DoublePendulumAnimator._trim_data(times, states, external_forces)

        anim = animation.FuncAnimation(
            fig,
            animate,
            interval=interval,
            frames=len(states),
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

    @staticmethod
    def _trim_data(times, states, forces):
        """
        Trims the data to match FPS (65).
        """
        num_frames = len(times)
        print(f"{times[0]} - {times[-1]} frames: {num_frames}")
        time_interval = times[-1] - times[0]
        fps = num_frames / time_interval

        while fps > 65:
            times = times[1::2]
            states = states[1::2]
            forces = forces[1::2]

            num_frames = len(times)
            time_interval = times[-1] - times[0]
            fps = num_frames / time_interval

        interval = 1000/fps
        return (times, states, forces, fps, interval)
