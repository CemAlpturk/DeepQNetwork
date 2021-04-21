import numpy as np
import time
import random
from os import path

from collections import deque
from matplotlib import pyplot as plt

from .NetworkBuilder import NetworkBuilder
from .Controller import Controller
#from Environments import PendulumOnCartEnvironment
#from Environments import PendulumOnCart

class QAgent:
    """
    Represents a Q-Agent.
    An agent that trains a network based on Q-Reinforcement Learning.
    """

    def __init__(
        self,
        environment,
        network_parameters : dict):
        """
        Initializes a Q-Agent.
        """

        self.environment = environment

        # For reshaping TODO: (check if it's possible to avoid reshaping in the algorithm)
        self.state_size = environment.state_size

        self.q_network = NetworkBuilder.Build(network_parameters)
        self.target_network = NetworkBuilder.Build(network_parameters)
        self._align_target_model()

        self.file_name = "scores.csv"
        file = open(self.file_name, "w")
        file.close()
        self.eval = []

        self.experience = deque(maxlen=2000)

    def train(
            self,
            max_episodes : int,
            exploration_rate=0.9,
            discount=0.9,
            batch_size=32,
            timesteps_per_episode=200,
            warm_start=False) -> Controller:
        """
        Trains the network with specified arguments.

        # TODO: Describe all arguments.

        returns Controller for controlling system provided by environment.

        # TODO: Fix and update the summary.

        # TODO: Discuss:
                * should there be a condition that stops training if the evaluation reaches a "good enough" value?
                * This can help avoid overtraining?

        # TODO: Add input validation.
        """

        # Load existing model for warm start
        if warm_start:
            check = self._load_model()
            if not check:
                print("Using default network") # TODO: temp solution

        for e in range(1, max_episodes + 1):
            t1 = time.time()
            total_reward = 0
            terminated = False
            t = 0
            state = self.environment.reset(random=True) # start from random state

            # TODO: Check if possible to avoid reshape!!
            state = state.reshape(1, self.state_size)

            for timestep in range(timesteps_per_episode):
                # Predict which action will yield the highest reward.
                action = self._act(state, exploration_rate)

                # Take the system forward one step in time.
                next_state = self.environment.step(action)

                # Compute the actual reward for the new state the system is in.
                time = timestep * self.environment.step_size
                reward = self.environment.reward(next_state, time)

                # Check whether the system has entered a terminal case.
                terminated = self.environment.terminated(next_state, time)

                # TODO: Can this be avoided?
                next_state = next_state.reshape(1, self.environment.state_size)

                # Store results for current step.
                self._store(state, action, reward, next_state, terminated)

                # Update statistics.
                total_reward += reward
                state = next_state
                t = timestep

                # Terminate episode if the system has reached a termination state.
                if terminated:
                    break

            if len(self.experience) > batch_size:
                self._experience_replay(batch_size, discount)
                exploration_rate *= 0.999

            t2 = time.time()
            print(
                f"Episode: {e:>5}, "
                f"Score: {total_reward:>10.1f}, "
                f"Steps: {t:>4}, "
                f"Simulation Time: {(t*self.environment.step_size):>6.2f} Seconds, "
                f"Computation Time: {(t2-t1):>6.2f} Seconds, "
                f"Exploration Rate: {exploration_rate:>0.3f}")

            if e % 100 == 0:
                self._align_target_model()
                self.environment.save(e)

            if e % 50 == 0:
                self._save_model()
                self._evaluate(10, max_steps=timesteps_per_episode)

        # Create Controller object
        controller = Controller(self.environment.get_action_space(), self.q_network)
        print("Controller Created")
        return controller

    def _act(
            self,
            state,
            exploration_rate : float):
        """
         Returns index
        """
        if np.random.rand() <= exploration_rate:
            return self.environment.get_random_action()

        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def _store(
            self,
            state,
            action,
            reward,
            next_state,
            terminated):
        """
        #TODO: Add summary.
        """

        self.experience.append((state, action, reward, next_state, terminated))

    def _experience_replay(self, batch_size, discount=0.9):
        """
        Updates network weights (fits model) with data stored in memory from
        executed simulations (training from experience).

        #TODO: Complete summary.
        """
        minibatch = random.sample(self.experience, batch_size)

        # TODO: The batch_size might not bee needed as an argument here if the reshape things can be resolved.
        states, actions, rewards, next_states, terminated = self._extract_data(batch_size, minibatch)
        targets = self._build_targets(batch_size, states, next_states, rewards, actions, terminated, discount)

        self.q_network.fit(states, targets, epochs=1, verbose=0)

    def _extract_data(self, batch_size, minibatch):
        """
        #TODO: Add summary and type description for variables.
        #
        # TODO: Complete summary.
        """

        # TODO: Extract the values into numpy arrays, could be done more efficient?
        states = np.array([x[0] for x in minibatch]).reshape(batch_size, self.state_size)
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch]).reshape(batch_size, self.state_size)
        terminated = np.array([x[4] for x in minibatch])

        return (states, actions, rewards, next_states, terminated)

    def _build_targets(
            self,
            batch_size,
            states,
            next_states,
            rewards,
            actions,
            terminated,
            discount):
        """
        TODO: Add summary.
        """

        i = terminated==0                                   # Steps that are not terminal
        targets = self.q_network.predict(states)            # Predict for each step
        t = self.target_network.predict(next_states[i, :])  # Predict for next steps that are not terminal

        targets[range(batch_size), actions] = rewards  # targets[:,action] = reward, selects the "action" column for each row
        targets[i, actions[i]] += discount * np.amax(t, axis=1) # add next estimation to non terminal rows

        return targets

    def _align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
        print("Target Network Realligned")

    def _evaluate(self, n, max_steps):
        """
        TODO: Add summary
        """

        print(f"Evaluating Model for {n} runs")
        # max_steps = 200
        total_reward = 0
        #time_step = 0.1
        for play in range(n):
            # TODO: Discuss - Shouldn't the reset function randomize the initial state
            # when evaluating? Need independent set to make a reasonable model evaluation.
            # Otherwise the reward and termination should be pretty much the same for every run?
            state = self.environment.reset(True).reshape(1, self.environment.state_size)

            for i in range(max_steps):
                # Determine the action to take based on the current state of the system.
                # TODO: Is this correct? The 'act' function actually uses a randomness to predict the action (when exploration rate is high)
                #       => It's not the network that predicts the action. We wan't to estimate the network here.
                action = self._act(state, -1)           # TODO: Discuss - using -1 to get around the random part of the '_act' method.
                #force = self.environmaction_space[action]

                # Take one step in time and apply the force from the action.
                next_state = self.environment.step(action)

                # Compute reward for the new state.
                reward = self.environment.reward(next_state)

                # Check wheter the new state is a termination or not.
                terminated = self.environment.terminated(next_state)

                # Update the current state variable.
                state = next_state.reshape(1, self.environment.state_size)

                # Update total reward for the play.
                total_reward += reward

                # Terminate if the termination condition is true.
                if terminated:
                    break

        average_reward = total_reward/n
        print(f"Average Total Reward: {average_reward:0.3f}")

        # Save to file
        self.eval.append(average_reward)
        file = open(self.file_name,"a")
        file.write(str(average_reward) + '\n')
        file.close()

        # Generate plot
        plt.plot(self.eval)
        plt.xlabel('Evaluations')
        plt.ylabel('Average Reward per Episode')
        plt.savefig('Scores.png')
        plt.close()

    def _save_model(self):
        print("Saving Model")
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





# TODO: Create dummy environment

if __name__ == "__main__":
    """
    Demonstrates how to use the Q-Agent class.
    """
    from Environments import PendulumOnCartEnvironment
    # Setup the environment (connects the problem to the q-agent).
    step_size = 0.02
    environment = PendulumOnCartEnvironment(step_size)

    # Setup Neural network parameters.
    from tensorflow.keras.optimizers import Adam
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
    training_episodes = 1
    controller = agent.train(max_episodes=training_episodes)

    # Simulate problem using the trained controller.
    max_time_steps = 100
    state = environment.reset()

    t = np.linspace(0, 10, 100)
    environment.solve(t, controller=controller.act)

    environment.animate()
