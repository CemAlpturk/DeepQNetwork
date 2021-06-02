import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
with tf.device("cpu:0"):
    from Agents import Controller
    from Environments import PendulumOnCartSimulator
    from Utilities.Animator import SinglePendulumAnimator

    action_space = [-10,0,10]
    filepath = "Logs/PendulumOnCart/2021-06-02_14-01-50/q_network"
    idx = [False,True,True,True]
    controller = Controller.load_controller(action_space, filepath, idx)

    max_angle = 0*np.pi/180
    initial_state = np.array([0.0,0.0,max_angle,0.0])

    problem_parameters = {
            "cart_mass": 1.0,
            "pendulum_mass": 0.1,
            "pendulum_length": 1.0,
        }


    problem = PendulumOnCartSimulator(problem_parameters, initial_state)

    t = np.linspace(0,10,500)
    problem.solve(t, controller=controller.act)

    # Pendulum settings.
    pendulum_settings = {
        "pendulum_length" : 1.,
        }

    # Plot settings.
    plot_settings = {
        "force_bar_show" : True,
        "force_action_space" : action_space,
    }

    SinglePendulumAnimator.animate_simulation(
        problem,
        pendulum_settings,
        plot_settings=plot_settings,
        save=True,
        output_filename="single.gif",
        title=f"Episode {0} - Inverted Double Pendulum on Cart",
        hide=False
    )
