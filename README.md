# FRTN70 — Project in Systems, Control and Learning!

This is a project in course FRTN70 at LTH, 2021.

Project is conducted by `Cem Alptürk` and `Pukashawar Pannu`.  
Project supervisor `Emil Vladu`.

## Goal
The goal of the project is to teach a controller to balance an inverted double pendulum on a cart using reinforcement learning.
This is chaotic problem and the controller is limited to a single degree of freedom for balancing the pendulums.

<div align="center">
  <figure>
    <img src="video/DoublePendulum_trained.gif">
    <figcaption>Trained agent controlling inverted double pendulum with initial conditions $\theta_1 = \theta_2 = -20^{\circ}$</figcaption>
  </figure>
</div>

## Get Started
Project has been developed using `Python 3`.
Dependencies can be installed using `pip`:

```bash
# Tensorflow 2.0 requires a pip version > 19.0
pip3 install --upgrade pip
pip3 install -r src/requirements.txt
```

A docker file has also been added that can be used to build and setup a docker container as environment for this project.
See the [Docker section](#docker-section) for more details.

Several python scripts have been added for simulating system with trained controller, for training a controller and for animating the training and/or animating the trained controller simulation.
Sections below describe the different scripts and how to use them.

### Simulate with Trained Controller
Scripts prefixed with `Play` are for simulating a system with a trained controller.
See `PlayDoublePendulum.py` and `PlayPendulum.py`.
Both scripts run a simulation with a trained controller and save the results in an animated `.gif` file and an `.mp4` file.
Run script with
```bash
python PlayDoublePendulum.py
```

See each script for settings regarding the simulation and animation.

### Train a Controller
Training is set up in scripts prefixed with `Training_`.
Each new call to the training script will create a folder structure under `Logs/<problem-name>/<date>` with outputs for the training process and the trained model.
Two versions of the model are saved after an evaluation period (or after specified number of episodes, see scripts for details), one for the `best` performing model and one for the `latest` model.
The `best` model is the one currently with the best evaluation performance and the latest one is after the latest fitted model.

```bash
python3 Training_Double_Pendulum.py
```

### Animate Results
Results can be animated into `.gif` and `.mp4` files using files prefixed with `Animate`.
The scripts will create a `.gif` file for every episode and then merge all `.gif`s to a single `.gif` called `total.gif`.
And finally convert `total.gif` into a `.mp4` file.
All files are outputted to the location of the training data and models.
See script for details on how to specify which training folder to animate.

```bash
python AnimateDoublePendulumGifs.py

# Assuming that the execution was for 2021-01-01_01-01-01
# The outputs can be found at
# Logs/DoublePendulumOnCart/2021-01-01_01-01-01/Episodes/<gifs>
```

### Docker <a name="docker-section"></a>
A dockerfile has been added to facilitate a docker environment with all requirements for running this project.
The dockerfile is in the root of the repository and can be built using:

```bash
docker build --tag frtn70_env . -f dockerfile
```

The tag can be changed to anything the user wants.

Created docker image can be launched in an interactive container.
Enable docker volumes to get an environment where the container and the host share files, this allows for a build environment in the container and a development environment on the host machine.
All files are also available on the host so that the user can view/save results easily.
Read more about docker volumes at [docker documentation](https://docs.docker.com/storage/volumes/).
Make sure to allow the docker runner to access the part of the hdd one wants to map inside the container.

```bash
# Volume can be setup as: -v <repo-path-on-host>:/<destination-in-container>
# where the first part <repo-path-on-host> is the path to the repository on the host machine
# and <destination-in-container> is where in the container the repo should be mapped
# The following command sets up a volume and runs bash in the container in interactive mode.
docker run -it -v <repo-path-on-host>:/<destination-in-container> <image-name> bash

# A full example where the host is a Windows machine and the repo in C:\repo¨
# and is mapped to /repo in the container:
docker run -it -v "c:\repo:/repo" frtn70_env bash
```

The interactive docker container gives the user a prompt and with the volume set up the entire repository will be available in `/repo` inside the container.

Two scripts have been added to simplify the `docker run` command.
One `.sh` script for Linux users and one `.ps1` script for Windows users.
The scripts should figure out the full path to the repository and set up the volume for the run command.
However, only the Windows script has been tested but the `.sh` script should help someone interested in automating the run command.

## Code Structure
The code has been separated in modules to simplify separation of logic.
`Environment` module contains everything related to the dynamic problem with the pendulum equations and a class structure that pans a way to apply an action to the system from the outside and also a way to receive a reward from the executed action.

Code in the `Agents` module contain the algorithms for training a controller using reinforcement learning.
The idea is that a trained algorithm produces a finished controller that can be applied and used directly on the dynamic system.

```python
# TODO: Add a short example for how to use an agent
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from Environments import DoublePendulumOnCartEnvironment
from Agents import QAgent

# Create Environment with default parameters
environment = DoublePendulumOnCartEnvironment()

# Create optimizer
optimizer = Adam()

# Specify network structure
# The last layer will have the same number of neurons as actions
layers = [(10,'relu'),(20,'relu'),(5,'linear')]

network_parameters = {
    "input_shape" : (6,),   # Number of states
    "layers": layers,       # Network structure
    "optimizer": optimizer,
    "loss_function": "mse",
    "initializer": tf.keras.initializers.he_uniform()   # Initialization strategy for the weights
}

# Create Agent
agent = QAgent(environment,network_parameters)

# Train the agent with default parameters and get the resulting controller 
controller = agent.train(max_episodes=200)


```

Code for training a controller for the double pendulum on a cart problem is found in `src/`.
With all dependencies installed it should be enough to run the `src/Training_Double_Pendulum.py` script.
The script produces training statistics and a trained controller that can be used to simulate a new pendulum scenario with the controller.

## Video
Video for training and final models simulated from different initial conditions for both the single and double pendulum problems can be found here ([.mp4](./video/Video.mp4) / [.mov](./video/Video.mov))
