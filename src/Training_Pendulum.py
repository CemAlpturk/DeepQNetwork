
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K

from Environments import PendulumOnCartEnvironment
from Agents import DoubleQAgent, QAgent
from Utilities.Animators import SinglePendulumAnimator

warm_start = False
# Check arguments
if len(sys.argv) > 1:
    if sys.argv[1] == "--warm":
        warm_start = True
    else:
        print("usage: Training_Pendulum.py --warm")

# Represents the maximum angle the pendulum is allowed
# to swing before it is considered out-of-bounds and
# the episode is terminated.
max_angle = 10*np.pi/180

def reward(state, t):
    """
    Reward function.
    Computes a reward based on the state of the system.

    reward = (max_angle - abs(theta)) / max_angle

    Args:
        states: array with the states of the system
                [x, xdot, theta, theta_dot]

        t:      time

    Returns:
        float:  the computed reward.
    """
    x, xdot, theta, theta_dot = state
    r_angle = (max_angle - abs(theta))/max_angle
    return r_angle

def terminated(state, t):
    """
    Termination function that determines whether the
    state of the system is a terminal state or not.

    State is terminal if the pendulum angle (theta)
    exceeds the maximum angle (max_angle) in either
    direction.

    Args:
        states: array with the states of the system
                [x, xdot, theta, theta_dot]

        t:      time

    Returns:
        bool:  True if the state is terminal, otherwise False.
    """
    x, xdot, theta, thetadot = state

    return abs(theta) > max_angle


# Setup the environment (connects the problem to the q-agent).
step_size = 0.02
environment = PendulumOnCartEnvironment(
        step_size=step_size,
        custom_reward_function=reward,
        custom_termination_function=terminated,
        action_space=[-10, 0, 10],
        lamb = max_angle*0.1)

# Setup Neural network parameters
optimizer = Adam(lr=0.001)

def custom_loss_function(y_true, y_pred):
    """
    Only the loss from the taken action affects the loss
    """
    # find the nonzero component of y_true
    idx = K.switch(K.not_equal(y_true, 0.0), y_pred, 0.0)
    loss = tf.subtract(y_true, idx)
    return K.square(K.sum(loss))

network_parameters = {
    "input_shape" : (4,),                                                      # Network input shape.
    "layers" : [(20, 'relu'), (40, 'relu'), (len(environment.get_action_space()), 'linear')],     # [(nodes, activation function)]
    "optimizer" : optimizer,                                                   # optimizer
    "loss_function" : custom_loss_function,                                    # loss function ('mse', etc.)
    "initializer" : tf.keras.initializers.he_uniform()
}

# States can be ignored from the input.
# Example: Ignoring the cart position can be achieved by: 'use_features = [False, True, True, True]'
# This can be useful for simplifying problems where it is known that some states won't
# contribute to solving the problem.
use_features = [False, True, True, True]

# Create agent.
agent = QAgent(environment, network_parameters, memory=2000, use_features=use_features)

# Train agent - produces a controller that can be used to control the system.
training_episodes = 200
controller = agent.train(
    max_episodes=training_episodes,
    timesteps_per_episode=200,
    batch_size=32,
    warm_start=warm_start,
    evaluate_model_period=10,
    evaluation_size=10,
    exploration_rate=0.5,
    exploration_rate_decay=0.99,
    model_alignment_period=10,
    discount=0.9,
    save_model_period=10,
    log_q_values=True)

# Simulate problem using the trained controller.
# TODO: Keep this or should it be encouraged to use ´Play´ script?
# state = environment.reset()

# t = np.linspace(0, 10, 50)
# environment.solve(t, controller=controller.act)
