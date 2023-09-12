from PIL import Image
import os
from torchvision.transforms import PILToTensor
import torch
from typing import TypedDict

class Keypoint(TypedDict):
    x: int
    y: int
    id: str
    idx: int
    
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

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return (PILToTensor()(img) / 255.0 - 0.5) * 2

def get_grid_keypoints(frame_w, frame_h, grid_size=30) -> list[Keypoint]:
        """     
        Returns:
            list[Keypoint]: list of grid keypoints
        """
        keypoints = []
        i = 0
        x_step = frame_w // grid_size
        y_step = frame_h // grid_size
        
        for x_id, x in enumerate(range(0, frame_w, x_step)):
            for y_id, y in enumerate(range(0, frame_h, y_step)):
                keypoints.append({
                    'x': x,
                    'y': y,
                    'id': f'{x_id}_{y_id}',
                    'idx': i
                })
                i += 1
        return keypoints