
from os import listdir
import re
import numpy as np

from Utilities.Animator import SinglePendulumAnimator
import imageio
import moviepy.editor as mp

# Folder containing gifs to animate.
root = "Logs/PendulumOnCart/2021-05-25_10-31-07/Episodes/"

# Pendulum settings.
pendulum_settings = { 
    "pendulum_length" : 1.,
    }

# Plot settings.
plot_settings = {
    "force_bar_show" : True,
    "force_action_space" : [40],
    "show_termination_boundary" : True,
    "termination_angle" : 10 * np.pi/180,
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

csv_files = [_ for _ in listdir(root) if _.endswith(".csv")]
csv_files.sort(key=natural_keys)

# Animate simulations.
# The .gif files are stored next to the .csv files.
for file in csv_files:
    episode = file.split("_")[1].split(".")[0]
    print(f"File: {file}, episode: {episode}")

    SinglePendulumAnimator.animate_from_csv(
        f'{root}{file}',
        pendulum_settings,
        plot_settings=plot_settings,
        save=True,
        title=f"Episode {episode} - Inverted Pendulum on Cart",
        output_filename=f"{root}{episode}.gif",
        hide=True
    )

    # Speed up gif to 60 fps.
    # gif = imageio.mimread(f"{root}{episode}_tmp.gif")
    # imageio.mimsave(f"{root}{episode}.gif", gif, fps=60)

# Merge gifs to a single gif.
gif_files = [_ for _ in listdir(root) if _.endswith(".gif")]
gif_files.sort(key=natural_keys)

new_gif = imageio.get_writer(f'{root}total.gif', fps=60)

for file in gif_files:
    print(f"Merging gif {file}.")
    loaded_gif = imageio.get_reader(f'{root}{file}')

    for iter in range(loaded_gif.get_length()):
        new_gif.append_data(loaded_gif.get_next_data())

    loaded_gif.close()

new_gif.close()

# Convert gif to mp4.
clip = mp.VideoFileClip(f"{root}total.gif")
clip.write_videofile(f"{root}total.mp4")
