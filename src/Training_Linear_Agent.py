

"""
Uses the available class structures to train a model
"""

import sys

import numpy as np


from Environments import DoublePendulumOnCartEnvironment
from Agents import LinearAgent


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
    x, theta1, theta2, xdot, theta1dot, theta2dot = state
    s = state[0:3]
    r = - np.dot(s,s)
    alive = 1
    return alive + r

def terminated(state, t):
    x, theta1, theta2, xdot, theta1dot, thteta2dot = state

    return abs(theta1) > max_angle


# Setup the environment (connects the problem to the q-agent).
step_size = 0.01
environment = DoublePendulumOnCartEnvironment(
        step_size=step_size,
        custom_reward_function=reward,
        custom_termination_function=terminated,
        action_space=[-40,-20,-10,-5,0,5,10,20,40],
        lamb=0.1*np.pi/180)



# Create agent.
agent = LinearAgent(environment)

# Train agent - produces a controller that can be used to control the system.
controller = agent.train(
        max_episodes=1000,
        timesteps_per_episode=300,
        warm_start=warm_start,
        evaluate_model_period=25,
        discount=0.1,
        exploration_rate=0.01,
        exploration_rate_decay=1,
        learning_rate=0.0001)

# Simulate problem using the trained controller.
state = environment.reset()

t = np.linspace(0, 10, 500)
environment.solve(t, controller=controller.act)

# environment.animate()
# environment.problem.animate(save=True,filename="resultDoublePendulum.gif")
