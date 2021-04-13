
import numpy as np

from tensorflow.keras.optimizers import Adam

from Environments import PendulumOnCartEnvironment
from Agents import QAgent


# Setup the environment (connects the problem to the q-agent).
step_size = 0.02
environment = PendulumOnCartEnvironment(step_size)

# Setup Neural network parameters.

optimizer = Adam(lr=0.02)
learning_rate = 0.01
network_parameters = {
    "input_shape" : (4,),                                       # Network input shape.
    "layers" : [(20, 'relu'), (40, 'relu'), (len(environment.get_action_space()), 'linear')],     # [(nodes, activation function)]
    "optimizer" : optimizer,                                    # optimizer
    "loss_function" : "mse",                                    # loss function ('mse', etc.)
}

# Create agent.
agent = QAgent(environment, network_parameters)

# Train agent - produces a controller that can be used to control the system.
training_episodes = 100
controller = agent.train(max_episodes=training_episodes)

# Simulate problem using the trained controller.
max_time_steps = 100
state = environment.reset()

t = np.linspace(0, 10, 100)
environment.solve(t, controller=controller.act)

environment.animate()
