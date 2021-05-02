
import sys

import numpy as np

from tensorflow.keras.optimizers import Adam

from Environments import PendulumOnCartEnvironment
from Agents import QAgent


warm_start = False
# Check arguments
if len(sys.argv) > 1:
    if sys.argv[1] == "--warm":
        warm_start = True
    else:
        print("usage: Training_Pendulum.py --warm")



max_angle = 20*np.pi/180

def reward(state, t):
    x, xdot, theta, thetadot = state
    r_angle = (max_angle - abs(theta))/max_angle
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
        lamb = max_angle)

# Setup Neural network parameters.

optimizer = Adam(lr=0.02)

network_parameters = {
    "input_shape" : (4,),                                       # Network input shape.
    "layers" : [(20, 'relu'), (40, 'relu'), (len(environment.get_action_space()), 'linear')],     # [(nodes, activation function)]
    "optimizer" : optimizer,                                    # optimizer
    "loss_function" : "mse",                                    # loss function ('mse', etc.)
}

# Create agent.
agent = QAgent(environment, network_parameters, memory=200)

# Train agent - produces a controller that can be used to control the system.
training_episodes = 250
controller = agent.train(
    max_episodes=training_episodes,
    batch_size=200,
    warm_start=warm_start)

# Simulate problem using the trained controller.
max_time_steps = 100
state = environment.reset()

t = np.linspace(0, 10, 100)
environment.solve(t, controller=controller.act)

environment.animate()
environment.problem.animate(save=True,filename="resultPendulum.gif")
