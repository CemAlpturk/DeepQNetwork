import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

import imageio
import moviepy.editor as mp
import ast
from pygifsicle import optimize

with tf.device("cpu:0"):
    from Agents import Controller
    from Environments import DoublePendulumOnCartSimulator
    from Utilities.Animators import DoublePendulumAnimator

    # action_space = [-3, -1, 0, 1, 3]
    action_space = [-40, -10, -5, -1, 0, 1, 5, 10, 40]
    filepath = "Final_models/DoublePendulum/2021-06-02_12-43-09/best/q_network"
    idx = [False,True,True,True,True,True]
    controller = Controller.load_controller(action_space, filepath, idx)

    max_angle = -20*np.pi/180
    outer_angle = -20*np.pi/180
    initial_state = np.array([2.0, max_angle, outer_angle, 0.0, 0.0, 0.0])

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

    t = np.linspace(0, 5, 250)
    problem.solve(t, controller=controller.act)
    u = problem.external_forces
    theta1 = [x[1]*(180/np.pi) for x in problem.states]
    theta2 = [x[2]*(180/np.pi) for x in problem.states]
    t = problem.time

    plt.plot(t,theta1)
    plt.plot(t,theta2)
    plt.axhline(y=15,linestyle="--",c="r")
    plt.axhline(y=-15,linestyle="--",c="r")
    plt.xlabel("Time [Seconds]")
    plt.ylabel("Angle [Degrees]")
    plt.title("Trained Controller")
    plt.legend([r"$\theta_1$",r"$\theta_2$","Training boundary"])
    plt.show()

    # (unique, counts) = np.unique(u, return_counts=True)
    # labels = [str(u) for u in unique]
    # plt.bar(labels,counts/np.sum(counts)*100)
    # plt.xlabel("Force [N]")
    # plt.ylabel("Usage [%]")
    # plt.title("Action Distribution")
    # plt.show()
    #
    # # Pendulum settings.
    # pendulum_settings = {
    #     "inner_pendulum_length" : 1.,
    #     "outer_pendulum_length" : 1.
    #     }
    #
    # # Plot settings.
    # plot_settings = {
    #     "force_bar_show" : True,
    #     "force_action_space" : action_space,
    #     "show_termination_boundary": True,
    #     "termination_angle_inner_pendulum": 15*np.pi/180,
    #     "termination_angle_outer_pendulum": 15*np.pi/180,
    # }
    #
    # DoublePendulumAnimator.animate_simulation(
    #     problem,
    #     pendulum_settings,
    #     plot_settings=plot_settings,
    #     save=True,
    #     output_filename="DoublePendulum_trained.gif",
    #     title=f"Trained Agent",
    #     hide=True
    # )
    #
    # clip = mp.VideoFileClip("DoublePendulum_trained.gif")
    # clip.write_videofile("DoublePendulum_trained.mp4")
