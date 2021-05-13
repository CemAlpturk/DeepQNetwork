

"""
Uses the available class structures to train a model
"""

import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from Environments import DoublePendulumOnCartEnvironment
from Agents import QAgent, DoubleQAgent


warm_start = False
# Check arguments
if len(sys.argv) > 1:
    if sys.argv[1] == "--warm":
        print("Warm start")
        warm_start = True
    else:
        print("usage: Training_Pendulum.py --warm")

max_angle = 10*np.pi/180

def custom_loss_function(y_true, y_pred):
    """
    Only the loss from the taken action affects the loss
    """
    # find the nonzero component of y_true
    idx = K.switch(K.not_equal(y_true, 0.0), y_pred, 0.0)
    loss = tf.subtract(y_true, idx)
    return K.square(K.sum(loss))

def reward(state, t):
    x, theta1, theta2, xdot, theta1dot, theta2dot = state
    #r_angle2 = (max_angle - abs(theta2))/max_angle
    xp = x + np.sin(theta1) + np.sin(theta2)
    yp = np.cos(theta1) + np.cos(theta2)
    dist_penalty = 0.01 * xp ** 2 + (yp - 2) ** 2
    #vel_penalty = 1e-3 * theta1dot**2 + 5e-3 * theta2dot**2
    #dist_penalty = xp ** 2 + (yp - 2) ** 2
    vel_penalty = theta1dot**2 + theta2dot**2
    alive_bonus = 10
    r = alive_bonus - dist_penalty - vel_penalty
    return r

def terminated(state, t):
    x, theta1, theta2, xdot, theta1dot, thteta2dot = state

    return abs(theta1) > max_angle or abs(theta2) > max_angle
    #yp = np.cos(theta1) + np.cos(theta2)
    #return yp <= 1.9

# Setup the environment (connects the problem to the q-agent).
step_size = 0.01
environment = DoublePendulumOnCartEnvironment(
        step_size=step_size,
        custom_reward_function=reward,
        custom_termination_function=terminated,
        action_space=[-10,0,10],
        lamb=0.1)

# Setup Neural network parameters.
initial_learning_rate = 0.1
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=1,
    staircase=True)
optimizer = Adam(learning_rate=lr_schedule)

layers = []
for _ in range(5):
    layers.append((20,'relu'))
layers.append((len(environment.action_space),'linear'))
network_parameters = {
    "input_shape" : (6,),                                       # Network input shape.
    "layers" : layers,                                          # [(nodes, activation function)]
    "optimizer" : optimizer,                                    # optimizer
    "loss_function" : custom_loss_function,                                    # loss function ('mse', etc.)
    "initializer" : tf.keras.initializers.he_uniform
}

# Create agent.
agent = DoubleQAgent(environment, network_parameters, memory=2000)

# Train agent - produces a controller that can be used to control the system.
controller = agent.train(
        max_episodes=10000,
        timesteps_per_episode=1000,
        warm_start=warm_start,
        evaluate_model_period=10,
        model_alignment_period=10,
        save_animation_period=25,
        batch_size=32,
        discount=0.9,
        exploration_rate=0.5,
        exploration_rate_decay=0.99,
        min_exploration_rate=0.1,
        save_model_period=10,
        epochs=1)

# Simulate problem using the trained controller.
state = environment.reset()

t = np.linspace(0, 10, 500)
environment.solve(t, controller=controller.act)

environment.animate()
environment.problem.animate(save=True,filename="resultDoublePendulum.gif")
