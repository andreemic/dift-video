from PIL import Image, ImageDraw
import os
from torchvision.transforms import PILToTensor
import torch
from typing import TypedDict
from performance import MockPerformanceManager
import cv2
import numpy as np

class Keypoint(TypedDict):
    x: int
    y: int
    id: str
    idx: int

def draw_keypoint_on_frame(frame: Image.Image, keypoint: Keypoint, color='red', radius=1.5, dynamic_color=True) -> Image.Image:
    """
    Draws a keypoint on a frame.

    Args:
        frame (Image.Image): PIL image frame
        keypoint (Keypoint): keypoint dictionary
        color (str): color of the keypoint
        radius (int): radius of the keypoint
        dynamic_color (bool): if True, the color of the keypoint will be based on the keypoint's x and y coordinates (x -> red, y -> green)
    
    Returns:
        Image.Image: PIL image frame with keypoint drawn on it
    """
    draw = ImageDraw.Draw(frame)
    
    # If dynamic color is True, map x and y coordinates to red and green channels.
    if dynamic_color:
        r = min(255, int(keypoint['x']))
        g = min(255, int(keypoint['y']))
        keypoint_color = (r, g, 0)  # Use RGB format with x mapped to red and y mapped to green.
    else:
        keypoint_color = color

    # Draw the circle on the image.
    left_up_point = (keypoint['x'] - radius, keypoint['y'] - radius)
    right_down_point = (keypoint['x'] + radius, keypoint['y'] + radius)
    draw.ellipse([left_up_point, right_down_point], fill=keypoint_color)

    return frame

def draw_keypoints_on_frames(frames: list[Image.Image], keypoints: list[list[Keypoint]], color='red', radius=1.5, dynamic_color=True, perf_manager=MockPerformanceManager) -> list[Image.Image]:
    new_frames = []
    for frame, frame_keypoints in zip(frames, keypoints):
        new_frame = frame.copy()
        for keypoint in frame_keypoints:
            perf_manager.start('draw_keypoint_on_frame')
            new_frame = draw_keypoint_on_frame(new_frame, keypoint, color=color, radius=radius, dynamic_color=dynamic_color)
            perf_manager.end('draw_keypoint_on_frame')
        new_frames.append(new_frame)
    return new_frames

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

def load_video_as_frames(fpath, fps=24):
    """
    Load an mp4 video as a list of PIL image frames.
    
    Parameters:
    - fpath: path to the video file
    - fps: desired frames per second to extract (default: 24)
    
    Returns:
    - List of PIL Image objects
    """
    frames = []

    # Open video using OpenCV
    cap = cv2.VideoCapture(fpath)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video at {fpath}")

    # Get the original video's frames per second
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Compute frame skip to match the desired fps
    frame_skip = int(original_fps / fps)

    # Read and append frames to the list
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only store every nth frame to match the desired fps
        if idx % frame_skip == 0:
            # Convert BGR to RGB format (OpenCV loads images in BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert OpenCV image to PIL image
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)

        idx += 1

    cap.release()

    return frames

def save_frames_as_video(frames, fpath, fps=24):
    """
    Save a list of PIL Image frames as an mp4 video.
    
    Parameters:
    - frames: List of PIL Image objects
    - fpath: Path where the video will be saved
    - fps: Desired frames per second for the output video (default: 24)
    
    Returns:
    None
    """
    
    # Check if there are frames
    if not frames:
        raise ValueError("No frames provided")
    
    # Convert the first frame to numpy to get the dimensions
    frame_np = np.array(frames[0])
    h, w, layers = frame_np.shape
    size = (w, h)
    
    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    out = cv2.VideoWriter(fpath, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        # Convert PIL Image to NumPy array
        frame_np = np.array(frame)
        
        # Convert RGB to BGR format (as OpenCV works with BGR for writing videos)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame_bgr)
    
    # Release the VideoWriter
    out.release()

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

