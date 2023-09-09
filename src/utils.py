from PIL import Image
import os

def load_frames(frames_dir):
    """
    Read in frames from a directory with files like 00001.jpg, 00002.jpg, etc.
    Supports .jpg and .png files.

    Returns:
        frames (list): list of PIL image frames
        fpaths (list): list of file paths to frames
    """

    frames = []
    fpaths = []
    for file in sorted(os.listdir(frames_dir)):
        if file.endswith(".jpg") or file.endswith(".png"):
            fpath = os.path.join(frames_dir, file)
            frames.append(Image.open(fpath))
            fpaths.append(fpath)
    return frames, fpaths