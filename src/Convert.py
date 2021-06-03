import moviepy.editor as mp

clip = mp.VideoFileClip("../video/Video.mp4")
clip.write_videofile("../video/Video.mov")
