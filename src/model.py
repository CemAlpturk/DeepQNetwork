import numpy as np
import random
#from IPython.display import clear_output
from collections import deque
import progressbar
# This is a test
import gym

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

import pandas as pd
from matplotlib import pyplot as plt
import time

enviroment = gym.make("CartPole-v0").env
#enviroment.render()

print('Number of states: {}'.format(enviroment.observation_space.shape))
print('Number of actions: {}'.format(enviroment.action_space.n))


class Agent:
    def __init__(self, enviroment, optimizer):

        # Initialize atributes
        self._state_size = enviroment.observation_space.shape[0]
        self._action_size = enviroment.action_space.n
        self._optimizer = optimizer

        self.expirience_replay = deque(maxlen=2000)

        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1

        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))

    def _build_compile_model(self):
        model = Sequential()
        #model.add(Embedding(self._state_size, 10, input_length=1))
        #model.add(Reshape((10,)))
        model.add(Dense(10, input_shape=(1,4), activation='relu'))
        model.add(Dense(10, activation='relu'))
        #model.add(Dense(10, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return enviroment.action_space.sample()

        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, next_state, terminated in minibatch:

            target = self.q_network.predict(state)

            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs=1, verbose=0)



scores = []
times = []
optimizer = Adam(learning_rate=0.01)
agent = Agent(enviroment, optimizer)

batch_size = 32
num_of_episodes = 100
timesteps_per_episode = 1000
agent.q_network.summary()
for e in range(0, num_of_episodes):
    t1 = time.time()
    # Reset the enviroment
    state = enviroment.reset()
    state = np.reshape(state, [1, 4])

    # Initialize variables
    total_reward = 0
    terminated = False

    bar = progressbar.ProgressBar(maxval=timesteps_per_episode / 10,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for timestep in range(timesteps_per_episode):
        # Run Action
        action = agent.act(state)

        # Take action
        next_state, reward, terminated, info = enviroment.step(action)
        next_state = np.reshape(next_state, [1, 4]) # TODO take shape from env
        agent.store(state, action, reward, next_state, terminated)

        state = next_state
        total_reward += reward
        if terminated:
            agent.alighn_target_model()
            break

        if len(agent.expirience_replay) > batch_size:
            agent.retrain(batch_size)

        if timestep % 10 == 0:
            bar.update(timestep / 10 + 1)

    bar.finish()
    t2 = time.time()
    print(f"Episode: {e}, Score: {total_reward}, Time: {(t2-t1):0.2f} Seconds")
    scores.append(total_reward)
    times.append(t2-t1)
    df = pd.DataFrame({'Scores':scores, 'Times':times})
    df.to_csv('Scores.csv')
    df.plot(y='Scores',use_index=True)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('Scores.png')
    plt.close()

    df.plot(y='Times',use_index=True)
    plt.xlabel('Episode')
    plt.ylabel('Time')
    plt.savefig('Times.png')
    plt.close()
    if (e + 1) % 10 == 0:
        print("**********************************")
        print("Episode: {}".format(e + 1))
        state = enviroment.reset()
        state = np.reshape(state,(1,4))
        max_step = 200
        i = 0
        terminated = False
        while not terminated  and i < max_step:
            action = agent.act(state)
            state, reward, terminated, info = enviroment.step(action)
            state = np.reshape(state, (1, 4))
            enviroment.render()
            i += 1
        enviroment.close()
        print("**********************************")
