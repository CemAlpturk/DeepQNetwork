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

    t = np.linspace(0,5,250)
    problem.solve(t, controller=controller.act)

    #
    # angle = [x[2]*(180/np.pi) for x in problem.states]
    # t = problem.time
    # plt.plot(t,angle)
    # plt.xlabel("Time [seconds]")
    # plt.ylabel("Angle [degrees]")
    # plt.title("Trained Controller")
    # plt.show()

    # Pendulum settings.
    pendulum_settings = {
        "pendulum_length" : 1.,
        }

    # Plot settings.
    plot_settings = {
        "force_bar_show" : True,
        "force_action_space" : action_space,
        "show_termination_boundary": True,
        "termination_angle": 10*np.pi/180
    }

    SinglePendulumAnimator.animate_simulation(
        problem,
        pendulum_settings,
        plot_settings=plot_settings,
        save=True,
        output_filename="single_trained.gif",
        title=f"Trained Agent",
        hide=False
    )

    # Convert gif to mp4.
    clip = mp.VideoFileClip("single_trained.gif")
    clip.write_videofile("single_trained.mp4")
