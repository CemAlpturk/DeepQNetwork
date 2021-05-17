import numpy as np
import time
import random
from os import path

from collections import deque
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

from .Controller import Controller
from Logger.Logger import Logger


class LinearAgent:
    """
    Represents a Q-Agent.
    An agent that trains a network based on Q-Reinforcement Learning.
    """

    def __init__(
        self,
        environment):
        """
        Initializes a Q-Agent.

        TODO: Explain all parameters.

        TODO: Add custom parameter for custom .csv scores (that are loaded for the evaluator).
        """

        self.environment = environment

        # For reshaping TODO: (check if it's possible to avoid reshaping in the algorithm)
        self.state_size = environment.state_size
        self.action_space = environment.action_space
        self.action_size = len(environment.action_space)
        print(self.action_space)

        self.w = np.zeros((36,1))
        self.w_target = np.zeros((36,1))
        self.poly = PolynomialFeatures(2)

        self.Logger = Logger(environment.name)
        self.memory = deque(maxlen=2000)


    def train(
            self,
            max_episodes : int,
            exploration_rate=0.9,
            discount=0.9,
            timesteps_per_episode=200,
            warm_start=False,
            save_animation_period=100,
            evaluate_model_period=50,
            model_alignment_period=50,
            evaluation_size=10,
            exploration_rate_decay=0.99,
            learning_rate=0.1) -> Controller:
        """
        Trains the network with specified arguments.

        # TODO: Describe all arguments.

        returns Controller for controlling system provided by environment.

        # TODO: Fix and update the summary.

        # TODO: Discuss:
                * should there be a condition that stops training if the evaluation reaches a "good enough" value?
                * This can help avoid overtraining?

        # TODO: Add input validation.

            # model_alignment_period (align models once after each period, period = n number of episodes)
            # save_animation_period (save animation once after each period, period = n number of episodes)
            # save_model_period = (save model once after each period, period = n number of episodes)
            # evaluate_model_period = (evaluate model once after each period, period = n number of episodes)
            # evaluation_size = (number of simulations to run to evaluate the model)
            # exploration_rate_decay = (how much the exploration rate should change after each episode)
        """

        # Load existing model for warm start
        if warm_start:
            check = self._load_model()
            if not check:
                print("Using default network") # TODO: temp solution

        for episode in range(1, max_episodes + 1):
            t1 = time.time()
            total_reward = 0
            terminated = False
            steps = 0
            state = self.environment.reset(random=True) # start from random state

            # TODO: Check if possible to avoid reshape!!
            state = state.reshape(1, self.state_size)


            for timestep in range(timesteps_per_episode):
                # Predict which action will yield the highest reward.
                # if np.random.rand() <= exploration_rate:
                #     action = self.environment.get_random_action()

                action = self._act(state, exploration_rate)

                # Take the system forward one step in time.
                next_state = self.environment.step(action)

                # Compute the actual reward for the new state the system is in.
                current_time = timestep * self.environment.step_size
                reward = self.environment.reward(next_state, current_time)

                # Check whether the system has entered a terminal case.
                terminated = self.environment.terminated(next_state, current_time)

                # TODO: Can this be avoided?
                next_state = next_state.reshape(1, -1)

                self.memory.append((state,action,reward,next_state))

                _ = self._update_weights(state, action, reward, next_state, discount, learning_rate)
                # Update statistics.
                total_reward += reward
                state = next_state
                steps = timestep+1



                # Terminate episode if the system has reached a termination state.
                if terminated:
                    break


            exploration_rate *= exploration_rate_decay
            t2 = time.time()
            print(
                f"Episode: {episode:>5}, "
                f"Score: {total_reward:>10.1f}, "
                f"Steps: {steps:>4}, "
                f"Simulation Time: {(steps * self.environment.step_size):>6.2f} Seconds, "
                f"Computation Time: {(t2-t1):>6.2f} Seconds, "
                f"Exploration Rate: {exploration_rate:>0.3f}")


            if episode % save_animation_period == 0:
                self.environment.save(episode)

            if episode % evaluate_model_period == 0:
                self._evaluate(evaluation_size, max_steps=timesteps_per_episode,episode=episode)

            if episode % model_alignment_period == 0:
                self._allign_model()

        # Create Controller object
        #controller = Controller(self.environment.get_action_space(), self.q_network)
        #print("Controller Created")
        #return controller

    def _act(
            self,
            state,
            exploration_rate : float):
        """
         Returns index
        """
        if np.random.rand() <= exploration_rate:
            return self.environment.get_random_action()

        q_values = np.zeros(self.action_size)
        for i,action in enumerate(self.action_space):
            inp = self.poly.fit_transform(np.append(state,action).reshape(1,-1))
            q_values[i] = np.dot(inp,self.w)[0]
        return np.argmax(q_values)
        #q_values = self.q_network.predict(state)
        #return np.argmax(q_values[0])

    # def _experience_replay(self,batch_size,gamma,alpha):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for elem in minibatch:
    #         state, action, reward, next_state = elem


    def _update_weights(self, state, action, reward, next_state, gamma, alpha):

        next_action = self._act(next_state,-1)
        state_inp = self.poly.fit_transform(np.append(state,self.action_space[action]).reshape(1,-1))
        next_state_inp = self.poly.fit_transform(np.append(next_state,self.action_space[next_action]).reshape(1,-1))
        q = np.dot(state_inp,self.w)
        next_q = np.dot(next_state_inp,self.w)
        delta = reward + gamma*next_q - q
        #print(next_q)
        self.w += alpha*delta*np.transpose(state_inp)
        return next_action

    def _allign_model(self):
        self.w_target = np.copy(self.w)

    def _evaluate(self, n, max_steps, episode):
        """
        TODO: Add summary
        """

        print(f"Evaluating Model for {n} runs")
        # max_steps = 200
        total_reward = 0
        #time_step = 0.1
        u = []
        rewards = []
        term = []
        t = []
        states = []
        actions = self.action_space
        for play in range(n):
            state = self.environment.reset(True).reshape(1, -1)

            for i in range(max_steps):
                # Determine the action to take based on the current state of the system.
                # TODO: Is this correct? The 'act' function actually uses a randomness to predict the action (when exploration rate is high)
                #       => It's not the network that predicts the action. We wan't to estimate the network here.
                action = self._act(state, -1)           # TODO: Discuss - using -1 to get around the random part of the '_act' method.

                # Take one step in time and apply the force from the action.
                next_state = self.environment.step(action)

                # Compute reward for the new state.
                current_time = i * self.environment.step_size
                reward = self.environment.reward(next_state, current_time)

                # Check wheter the new state is a termination or not.
                terminated = self.environment.terminated(next_state, current_time)

                 #Log play
                if play == 0:
                    u.append(actions[action])
                    t.append(current_time)
                    rewards.append(reward)
                    term.append(terminated)
                    states.append(state)

                # Update the current state variable.
                state = next_state.reshape(1, -1)

                # Update total reward for the play.
                total_reward += reward


                # Terminate if the termination condition is true.
                if terminated:
                    break

        average_reward = total_reward/n
        print(f"Average Total Reward: {average_reward:0.3f}")

        # Log the recorded play
        self.Logger.log_episode(states,u,rewards,term,t,episode)


        self.Logger.log_eval(episode, average_reward)


    def _save_model(self):
        pass
        print("Saving Model")

        # TODO: Fix path for windows.
        filepath = f"./Models/{self.environment.name}/q_network"
        self.q_network.save(filepath)

    def _load_model(self):
        """
        Load pre-trained model for warm start
        """
        filepath = f"Models/{self.environment.name}/q_network"
        # Check if model exists in default directory
        if path.exists(filepath):
            self.q_network = NetworkBuilder._load_model(filepath)
            self.target_network = NetworkBuilder._load_model(filepath)
            print("Models loaded")
            return True
        else:
            print(f"'{filepath}' not found")
            return False
