import moviepy.editor as mp

clip = mp.VideoFileClip("output.gif")
clip.write_videofile("merged.mp4")
