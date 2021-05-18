import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
with tf.device("cpu:0"):
    from Agents import Controller
    from Environments import PendulumOnCartSimulator

    action_space = [-10,0,10]
    parent_dir = os.getcwd()
    filepath = os.path.join("Models","PendulumOnCart","q_network")
    #filepath = "Models/PendulumOnCart/q_network"
    controller = Controller.load_controller(action_space, filepath)

    max_angle = 15*np.pi/180
    initial_state = np.array([0.0,0.0,max_angle,0.0])

    problem_parameters = {
        "cart_mass": 1.0,
        "pendulum_mass": 0.1,
        "pendulum_length" : 1.0,
    }

    problem = PendulumOnCartSimulator(problem_parameters, initial_state)

    t = np.linspace(0,5,250)
    problem.solve(t, controller=controller.act)

    problem.animate()

    # Access data
    # states = problem.states
    #
    # df = pd.DataFrame(states,columns=['x','xdot','theta','thetadot'])
    #
    # theta = df['theta']*180/np.pi
    # t = problem.time
    #
    # plt.plot(t,theta)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Pendulum Angle [deg]")
    # plt.title("Trained Controller")
    # plt.show()
