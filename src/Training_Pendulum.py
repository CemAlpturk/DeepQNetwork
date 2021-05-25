
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K


from Environments import PendulumOnCartEnvironment
from Agents import DoubleQAgent, QAgent


warm_start = False
# Check arguments
if len(sys.argv) > 1:
    if sys.argv[1] == "--warm":
        warm_start = True
    else:
        print("usage: Training_Pendulum.py --warm")



max_angle = 10*np.pi/180

def reward(state, t):
    x, xdot, theta, thetadot = state
    r_angle = (max_angle - abs(theta))/max_angle
    # alive_bonus = 10
    # px = x + np.cos(theta)
    # py = np.sin(theta)
    # pos_penalty = px**2 + (1-py)**2
    # vel_penalty = thetadot**2
    # return alive_bonus - pos_penalty - vel_penalty
    return r_angle

def terminated(state, t):
    x, xdot, theta, thetadot = state

    return abs(theta) > max_angle


# Setup the environment (connects the problem to the q-agent).
step_size = 0.02
environment = PendulumOnCartEnvironment(
        step_size=step_size,
        custom_reward_function=reward,
        custom_termination_function=terminated,
        action_space=[-10,0,10],
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
    "input_shape" : (4,),                                       # Network input shape.
    "layers" : [(20, 'relu'), (40, 'relu'), (len(environment.get_action_space()), 'linear')],     # [(nodes, activation function)]
    "optimizer" : optimizer,                                    # optimizer
    "loss_function" : custom_loss_function,                                    # loss function ('mse', etc.)
    "initializer" : tf.keras.initializers.he_uniform()
}

# Create agent.
agent = QAgent(environment, network_parameters, memory=2000)

# Train agent - produces a controller that can be used to control the system.
training_episodes = 150
controller = agent.train(
    max_episodes=training_episodes,
    timesteps_per_episode=200,
    batch_size=32,
    warm_start=warm_start,
    evaluate_model_period=10,
    exploration_rate=0.5,
    exploration_rate_decay=0.99,
    model_alignment_period=10,
    discount=0.1,
    save_model_period=10,
    log_q_values=True)

# Simulate problem using the trained controller.
max_time_steps = 100
state = environment.reset()

t = np.linspace(0, 10, 50)
environment.solve(t, controller=controller.act)

environment.animate()
environment.problem.animate(save=True,filename="resultPendulum.gif")
