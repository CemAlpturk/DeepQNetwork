
from os import listdir
import re
import numpy as np

from Utilities.Animators import DoublePendulumAnimator
import imageio
import moviepy.editor as mp
import ast
from pygifsicle import optimize

# Folder containing gifs to animate.
<<<<<<< HEAD
base_root = "Final_models/DoublePendulum/2021-06-02_12-43-09/"
=======
# base_root = "Final_models/DoublePendulum/2021-06-01_14-29-16/"
base_root = "Logs/DoublePendulumOnCart/2021-06-11_06-53-59/"
>>>>>>> bbad9bdf7d8852ada8184bc79b183561375d3a74
root_episode = f"{base_root}Episodes/"

# Pendulum settings.
pendulum_settings = {
    "inner_pendulum_length" : 1.,
    "outer_pendulum_length" : 1.
    }


# Read settings.
with open(f"{base_root}params.txt") as f:
    raw_settings = f.read()

settings = ast.literal_eval(raw_settings)

# Plot settings.
plot_settings = {
    "force_bar_show" : True,
    "force_action_space" : settings["action_space"],
    "show_termination_boundary": True,
    "termination_angle_inner_pendulum": 15*np.pi/180,
    "termination_angle_outer_pendulum": 15*np.pi/180,
}

# Sort files based on episode number.
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

csv_files = [_ for _ in listdir(root_episode) if _.endswith(".csv")]
csv_files.sort(key=natural_keys)

# Animate simulations.
# The .gif files are stored next to the .csv files.
list = [10,20,50,100,150,200,250,500,750,1000,1500,2000,2030]
for file in csv_files:
    episode = file.split("_")[1].split(".")[0]
    if int(episode) not in list:
        continue
    print(f"File: {file}, episode: {episode}")

    DoublePendulumAnimator.animate_from_csv(
        f'{root_episode}{file}',
        pendulum_settings,
        plot_settings=plot_settings,
        save=True,
        title=f"Episode {episode} - Inverted Double Pendulum on Cart",
        output_filename=f"{root_episode}{episode}.gif",
        hide=True
    )

# Merge gifs to a single gif.
gif_files = [_ for _ in listdir(root_episode) if _.endswith(".gif")]
gif_files.sort(key=natural_keys)

new_gif = imageio.get_writer(f'{root_episode}total.gif', fps=60)

for file in gif_files:
    print(f"Merging gif {file}.")
    loaded_gif = imageio.get_reader(f'{root_episode}{file}')

    for iter in range(loaded_gif.get_length()):
        new_gif.append_data(loaded_gif.get_next_data())

    loaded_gif.close()

new_gif.close()

# Optimize gif
optimize(f'{root_episode}total.gif', f'{root_episode}total_optimized.gif')

# Convert gif to mp4.
clip = mp.VideoFileClip(f"{root_episode}total.gif")
clip.write_videofile(f"{root_episode}total.mp4")
