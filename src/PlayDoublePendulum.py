import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
with tf.device("cpu:0"):
    from Agents import Controller
    from Environments import DoublePendulumOnCartSimulator

    action_space = [-1,0,1]
    filepath = "Models/DoublePendulumOnCart/q_network"
    idx = [False,True,True,True,True,True]
    controller = Controller.load_controller(action_space, filepath, idx)

    max_angle = 0*np.pi/180
    initial_state = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

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

    t = np.linspace(0,5,250)
    problem.solve(t, controller=controller.act)

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


problem.animate()
