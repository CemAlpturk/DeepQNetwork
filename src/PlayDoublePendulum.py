import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
with tf.device("cpu:0"):
    from Agents import Controller
    from Environments import DoublePendulumOnCartSimulator
    from Utilities.Animator import DoublePendulumAnmiator

    action_space = [-1,0,1]
    filepath = "Models/DoublePendulumOnCart/q_network"
    idx = [False,True,True,True,True,True]
    controller = Controller.load_controller(action_space, filepath, idx)

    max_angle = 0*np.pi/180
    initial_state = np.array([0.0,0.01,0.0,0.0,0.0,0.0])

    problem_parameters = {
            "cart_mass": 1.0,
            "pendulum_1_mass": 0.1,
            "pendulum_2_mass": 0.1,
            "pendulum_1_length": 1.0,
            "pendulum_2_length": 1.0,
            "cart_friction" : 0.0,
            "pendulum_1_friction" : 0.0,
            "pendulum_2_friction" : 0.0
        }


    problem = DoublePendulumOnCartSimulator(problem_parameters, initial_state)

    t = np.linspace(0,5,500)
    problem.solve(t, controller=controller.act)

    # Pendulum settings.
    pendulum_settings = {
        "inner_pendulum_length" : 1.,
        "outer_pendulum_length" : 1.
        }

    # Plot settings.
    plot_settings = {
        "force_bar_show" : True,
        "force_action_space" : [-1,0,1],
    }

    DoublePendulumAnmiator.animate_simulation(
        problem,
        pendulum_settings,
        plot_settings=plot_settings,
        save=False,
        title=f"Episode {0} - Inverted Double Pendulum on Cart",
        hide=False
    )

    #problem.animate()


# import numpy as np
# from Environments import DoublePendulumOnCartSimulator
#
#
#
# max_angle = 0*np.pi/180
# initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#
# problem_parameters = {
#         "cart_mass": 1.0,
#         "pendulum_1_mass": 0.1,
#         "pendulum_2_mass": 0.1,
#         "pendulum_1_length": 1.0,
#         "pendulum_2_length": 1.0,
#         "cart_friction" : 0.0,
#         "pendulum_1_friction" : 0.0,
#         "pendulum_2_friction" : 0.0
#     }
#
# problem = DoublePendulumOnCartSimulator(problem_parameters, initial_state)
#
# t = np.linspace(0,2,200)
# problem.solve(t)
