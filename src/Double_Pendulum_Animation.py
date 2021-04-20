from Environments import DoublePendulumOnCartSimulator
import numpy as np

from Utilities.Animators import animate_multiple_double_pendulums

# Pendulum parameters:
problem_parameters = {
    "cart_mass": 1.0,
    "pendulum_1_mass": 0.1,
    "pendulum_2_mass": 0.1,
    "pendulum_1_length": 1.0,
    "pendulum_2_length": 1.0
}

# Pendulum 1
# [x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot]
x0_pendulum_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.060, 0.0])
pendulum_1 = DoublePendulumOnCartSimulator(problem_parameters, x0_pendulum_1)


# Pendulum 2
# [x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot]
x0_pendulum_2 = np.array([0.0, 0.0, 0.0, 0.0, 0.048, 0.0])
pendulum_2 = DoublePendulumOnCartSimulator(problem_parameters, x0_pendulum_2)

# Pendulum 3
# [x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot]
x0_pendulum_3 = np.array([0.0, 0.0, 0.0, 0.0, 0.052, 0.0])
pendulum_3 = DoublePendulumOnCartSimulator(problem_parameters, x0_pendulum_3)

# Pendulum 4
# [x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot]
x0_pendulum_4 = np.array([0.0, 0.0, 0.0, 0.0, 0.055, 0.0])
pendulum_4 = DoublePendulumOnCartSimulator(problem_parameters, x0_pendulum_4)

# Simulate problems:
t = np.linspace(0, 10, 500)

pendulum_1.solve(t)
pendulum_2.solve(t)
pendulum_3.solve(t)
# pendulum_4.solve(t)

# Add results for both pendulums.
pendulums = [pendulum_1, pendulum_2, pendulum_3]

# Animate both pendulums.
animate_multiple_double_pendulums(pendulums, "test.gif", True)
