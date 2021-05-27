
from os import listdir
import re

from Utilities.Animator import DoublePendulumAnmiator

# Folder containing gifs to animate.
root = "Logs/DoublePendulumOnCart/2021-05-06_07-25-38/Episodes/"

# Pendulum settings.
pendulum_settings = { 
    "inner_pendulum_length" : 1.,
    "outer_pendulum_length" : 1.
    }

# Plot settings.
plot_settings = {
    "force_bar_show" : True,
    "force_action_space" : [40],
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

files = [_ for _ in listdir(root) if _.endswith(".csv")]
files.sort(key=natural_keys)

# Animate simulations.
# The .gif files are stored next to the .csv files.
for file in files:
    episode = file.split("_")[1].split(".")[0]
    print(f"File: {file}, episode: {episode}")

    DoublePendulumAnmiator.animate_from_csv(
        f'{root}{file}',
        pendulum_settings,
        plot_settings=plot_settings,
        save=True,
        title=f"Episode {episode} - Inverted Double Pendulum on Cart",
        output_filename=f"{root}{episode}.gif",
        hide=True
    )
