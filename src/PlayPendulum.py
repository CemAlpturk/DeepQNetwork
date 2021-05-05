import numpy as np

from Agents import Controller
from Environments import PendulumOnCartSimulator

action_space = [-20,0,20]
filepath = "Models/PendulumOnCart/q_network"
controller = Controller.load_controller(action_space, filepath)

max_angle = 30*np.pi/180
initial_state = np.array([0.0,0.0,max_angle,0.0])

problem_parameters = {
    "cart_mass": 1.0,
    "pendulum_mass": 0.1,
    "pendulum_length" : 1.0,
}

problem = PendulumOnCartSimulator(problem_parameters, initial_state)

t = np.linspace(0,10,500)
problem.solve(t, controller=controller.act)

problem.animate()
