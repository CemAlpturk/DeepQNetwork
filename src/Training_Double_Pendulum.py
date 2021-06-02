

"""
Uses the available class structures to train a model
"""

import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
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

max_angle = 15*np.pi/180

def custom_loss_function(y_true, y_pred):
    """
    Only the loss from the taken action affects the loss
    """
    # find the nonzero component of y_true
    idx = K.switch(K.not_equal(y_true, 0.0), y_pred, 0.0)
    loss = tf.subtract(y_true, idx)
    return K.square(K.sum(loss))

def reward(state, t):
    if terminated(state, t):
        return -10

    x, theta1, theta2, xdot, theta1dot, theta2dot = state
    #r_angle2 = (max_angle - abs(theta2))/max_angle
    #r_angle1 = (max_angle - abs(theta1))/max_angleS
    #xp = x + np.sin(theta1) + np.sin(theta2)
    #yp = np.cos(theta1) + np.cos(theta2)
    #dist_penalty = 0.01 * xp ** 2 + (yp - 2) ** 2
    #vel_penalty = 1e-3 * theta1dot**2 + 5e-3 * theta2dot**2
    #dist_penalty = xp ** 2 + (yp - 2) ** 2
    #vel_penalty = theta1dot**2 + theta2dot**2
    angle_penalty = abs(theta1)/max_angle + abs(theta2)/max_angle
    vel_penalty = theta1dot**2 + theta2dot**2
    alive_bonus = 1
    r = alive_bonus - angle_penalty - vel_penalty
    #r = r_angle1**2 + r_angle2**2 -vel_penalty
    return max(0, r)

def terminated(state, t):
    x, theta1, theta2, xdot, theta1dot, thteta2dot = state

    return abs(theta1) > max_angle or abs(theta2) > max_angle
    #yp = np.cos(theta1) + np.cos(theta2)
    #return yp <= 1.9

# Setup the environment (connects the problem to the q-agent).
step_size = 0.02
environment = DoublePendulumOnCartEnvironment(
        step_size=step_size,
        custom_reward_function=reward,
        custom_termination_function=terminated,
        action_space=[-40, -10, -5, -1, 0, 1, 5, 10, 40],
        lamb=10*np.pi/180)

# Setup Neural network parameters.
initial_learning_rate = 0.00005
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=1,
    staircase=True)
optimizer = Adam(learning_rate=lr_schedule)

nodes = [30, 20, 40, 30, 30, 40, 30, 60, 60, 100, 40, 30, 70, 70, 70, 30, 30, 30, 30, 30]
layers = []
# for _ in range(20):
#     layers.append((30,'relu'))

for index in range(len(nodes)):
    layers.append((nodes[index], 'relu'))

layers.append((len(environment.action_space),'linear'))
network_parameters = {
    "input_shape" : (6,),                                # Network input shape.
    "layers" : layers,                                   # [(nodes, activation function)]
    "optimizer" : optimizer,                             # optimizer
    "loss_function" : custom_loss_function,              # loss function ('mse', etc.)
    "initializer" : tf.keras.initializers.he_uniform()
}

use_features = [True]*6
use_features[0] = False

# Create agent.
agent = QAgent(environment,
        network_parameters,
        memory=2000,
        use_features=use_features)

# Train agent - produces a controller that can be used to control the system.
controller = agent.train(
        max_episodes=10000,
        timesteps_per_episode=500,
        warm_start=warm_start,
        evaluate_model_period=10,
        model_alignment_period=1,
        save_animation_period=25,
        batch_size=32,
        discount=0.99,
        exploration_rate=0.9,
        exploration_rate_decay=0.995,
        min_exploration_rate=0.1,
        save_model_period=10,
        epochs=1,
        log_q_values=True)

# # Simulate problem using the trained controller.
# state = environment.reset()

# t = np.linspace(0, 10, 500)
# environment.solve(t, controller=controller.act)

# environment.animate()
# environment.problem.animate(save=True,filename="resultDoublePendulum.gif")
