import numpy as np

from Environments import PendulumOnCartSimulator
from Utilities.Animators import animate_multiple_single_pendulums

# Pendulum parameters
pendulum_parameters = {
    "cart_mass": 1.0,
    "pendulum_mass": 0.1,
    "pendulum_length" : 1.0,
}

# Create pendulums
pendulum_1_x0 = np.array([0.0, 0.0, 0.3, 0.06])
pendulum_1 = PendulumOnCartSimulator(pendulum_parameters, pendulum_1_x0)

pendulum_2_x0 = np.array([0.0, 0.0, 0.3, 0.08])
pendulum_2 = PendulumOnCartSimulator(pendulum_parameters, pendulum_2_x0)

pendulum_3_x0 = np.array([0.0, 0.0, 0.3, 0.08])
pendulum_3 = PendulumOnCartSimulator(pendulum_parameters, pendulum_3_x0)

# Simulate pendulums
t = np.linspace(0, 10, 500)

pendulum_1.solve(t)
pendulum_2.solve(t)
pendulum_3.solve(t)

# Animate
pendulums = [pendulum_1, pendulum_2, pendulum_3]
animate_multiple_single_pendulums(pendulums, "3_single_pendulums.gif", True)
