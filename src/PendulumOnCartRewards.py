from matplotlib import pyplot as plt
import numpy as np

from Environments import PendulumOnCartEnvironment


env = PendulumOnCartEnvironment()

angles = np.linspace(-15*np.pi/180, 15*np.pi/180, 100)

rewards = np.array([env.reward(np.array([0,0,angle,0])) for angle in angles])

plt.plot(angles,rewards)
plt.show()
