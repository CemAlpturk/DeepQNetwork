from os import listdir
import re
import imageio

# Folder containing gifs to animate.
root = "Logs/PendulumOnCart/2021-05-25_10-31-07/Episodes/"

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

files = [_ for _ in listdir(root) if _.endswith(".gif")]
files.sort(key=natural_keys)

#Create writer object
new_gif = imageio.get_writer('output.gif', fps=60)

for file in files:
    print(f"Merging gif {file}.")
    loaded_gif = imageio.get_reader(f'{root}{file}')

    for iter in range(loaded_gif.get_length()):
        new_gif.append_data(loaded_gif.get_next_data())

    loaded_gif.close()

new_gif.close()

