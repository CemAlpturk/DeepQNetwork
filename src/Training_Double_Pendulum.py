

"""
Uses the available class structures to train a model
"""

import numpy as np
from double_pendulum_inherited import DoublePendulumOnCart
from Connector2 import Connector2
from QAgent import QAgent
from Controller import Controller

if __name__ == "__main__":
    """
    Demonstration for training a double pendulum controller
    """

    problem_parameters = {
        "cart_mass": 1.0,
        "pendulum_1_mass": 0.1,
        "pendulum_2_mass": 0.1,
        "pendulum_1_length": 1.0,
        "pendulum_2_length": 1.0
    }
    initial_state = np.zeros(6)
    env = DoublePendulumOnCart(problem_parameters, initial_state)
    env.print()

    step_size = 0.02
    connector = Connector2(env, step_size)

    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(lr=0.01)
    network_parameters = {
        "input_shape": (6,),
        "layers": [(40, 'relu'), (40, 'relu'), (7, 'linear')],
        "optimizer": optimizer,
        "loss_function": "mse"
    }
    action_space = [-10,-5,-1,0,1,5,10]
    training_episodes = 1000
    agent = QAgent2(connector, network_parameters, action_space, training_episodes)

    controller = agent.train()
    state = connector.reset()

    t = np.linspace(0,10,100)
    env.solve(t, controller=controller.act)
    env.animate()
