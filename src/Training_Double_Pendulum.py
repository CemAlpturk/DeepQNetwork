

"""
Uses the available class structures to train a model
"""

import sys

import numpy as np

from tensorflow.keras.optimizers import Adam

from Environments import DoublePendulumOnCartEnvironment
from Agents import QAgent


warm_start = False
# Check arguments
if len(sys.argv) > 1:
    if sys.argv[1] == "--warm":
        print("Warm start")
        warm_start = True
    else:
        print("usage: Training_Pendulum.py --warm")

max_angle = 10*np.pi/180

def reward(state, t):
    x, theta1, theta2, xdot, theta1dot, thteta2dot = state
    r_angle2 = (max_angle - abs(theta2))/max_angle
    return r_angle2

def terminated(state, t):
    x, theta1, theta2, xdot, theta1dot, thteta2dot = state

    return abs(theta1) > max_angle or abs(theta2) > max_angle

# Setup the environment (connects the problem to the q-agent).
step_size = 0.02
environment = DoublePendulumOnCartEnvironment(
        step_size=step_size,
        custom_reward_function=reward,
        custom_termination_function=terminated,
        action_space=[-10,-5,0,5,10])

# Setup Neural network parameters.

optimizer = Adam(lr=0.01)
network_parameters = {
    "input_shape" : (6,),                                       # Network input shape.
    "layers" : [(100, 'relu'), (100, 'relu'), (len(environment.get_action_space()), 'linear')],     # [(nodes, activation function)]
    "optimizer" : optimizer,                                    # optimizer
    "loss_function" : "mse",                                    # loss function ('mse', etc.)
}

# Create agent.
agent = QAgent(environment, network_parameters, memory=200)

# Train agent - produces a controller that can be used to control the system.
controller = agent.train(
        max_episodes=100,
        warm_start=warm_start,
        batch_size=200)

# Simulate problem using the trained controller.
state = environment.reset()

t = np.linspace(0, 10, 500)
environment.solve(t, controller=controller.act)

environment.animate()
environment.problem.animate(save=True,filename="result.gif")
