from PIL import Image, ImageDraw, ImageStat, ImageColor
import colorsys

import os
from torchvision.transforms import PILToTensor
import torch
from typing import TypedDict
from performance import MockPerformanceManager
import cv2
import numpy as np
import subprocess
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures


class Keypoint(TypedDict):
    x: int
    y: int
    id: str
    idx: int


def get_opposite_color(color):
    """Returns the opposite color in HLS space."""
    r, g, b = color
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    
    h = (h + 0.5) % 1.0  # Rotate the hue by 180 degrees.
    l = 1.0 - l  # Invert the lightness.
    s = max(s, 0.8)  # Ensure high saturation, set it to at least 0.8

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))

def get_avg_color(image, box):
    """Get average color of a specific area in the image."""
    region = image.crop(box)
    stat = ImageStat.Stat(region)
    return tuple(map(int, stat.mean))


def draw_keypoint_on_frame(frame: Image.Image, keypoint: Keypoint, color='red', radius=1.5, dynamic_color=True, style='circle' or 'cross') -> Image.Image:
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
    if color == "auto":
        box = (
            keypoint['x'] - radius * 2, 
            keypoint['y'] - radius * 2, 
            keypoint['x'] + radius * 2, 
            keypoint['y'] + radius * 2
        )
        avg_color = get_avg_color(frame, box)
        keypoint_color = get_opposite_color(avg_color)
    elif dynamic_color:
        r = min(255, int(keypoint['x']))
        g = min(255, int(keypoint['y']))
        keypoint_color = (r, g, 0)
    else:
        keypoint_color = color


    if style == 'circle':
        # Draw the circle on the image.
        left_up_point = (keypoint['x'] - radius, keypoint['y'] - radius)
        right_down_point = (keypoint['x'] + radius, keypoint['y'] + radius)
        draw.ellipse([left_up_point, right_down_point], fill=keypoint_color)

    elif style == 'cross':
        cross_length = radius  # Adjust this value based on desired cross length.
        
        # Horizontal line of the cross.
        start_horizontal = (keypoint['x'] - cross_length, keypoint['y'])
        end_horizontal = (keypoint['x'] + cross_length, keypoint['y'])
        
        # Vertical line of the cross.
        start_vertical = (keypoint['x'], keypoint['y'] - cross_length)
        end_vertical = (keypoint['x'], keypoint['y'] + cross_length)
        
        draw.line([start_horizontal, end_horizontal], fill=keypoint_color, width=int(radius/2))
        draw.line([start_vertical, end_vertical], fill=keypoint_color, width=int(radius/2))
    else:
        raise ValueError(f"Unknown style {style}")

    return frame, keypoint_color

def draw_keypoints_on_frames(frames: list[Image.Image], keypoints: list[list[Keypoint]], color='red', radius=1.5, dynamic_color=True, perf_manager=MockPerformanceManager) -> list[Image.Image]:
    new_frames = []
    for frame, frame_keypoints in zip(frames, keypoints):
        new_frame = frame.copy()
        for keypoint in frame_keypoints:
            perf_manager.start('draw_keypoint_on_frame')
            color = keypoint.get('color', color)
            new_frame, kp_color = draw_keypoint_on_frame(new_frame, keypoint, color=color, radius=radius, dynamic_color=dynamic_color)
            perf_manager.end('draw_keypoint_on_frame')
        new_frames.append(new_frame)
    return new_frames
def load_frame(file, frames_dir):
    """
    Load a single frame.
    """
    fpath = os.path.join(frames_dir, file)
    return Image.open(fpath), fpath

def load_frames(frames_dir, max_workers=None):
    """
    Read in frames from a directory with files like 00001.jpg, 00002.jpg, etc.
    Supports .jpg and .png files.

    Returns:
        frames (list): list of PIL image frames
        fpaths (list): list of file paths to frames
    """
    frames = []
    fpaths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file in sorted(os.listdir(frames_dir)):
            if file.endswith(".jpg") or file.endswith(".png"):
                futures.append(executor.submit(load_frame, file, frames_dir))
        
        for future in concurrent.futures.as_completed(futures):
            frame, fpath = future.result()
            frames.append(frame)
            fpaths.append(fpath)
    
    # Return frames and paths in the correct order
    frames, fpaths = zip(*sorted(zip(frames, fpaths), key=lambda x: x[1]))
    return list(frames), list(fpaths)



def load_video_as_frames(fpath, fps=24, max_dimension=None):
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
    if frame_skip == 0:
        frame_skip = 1

    # Read and append frames to the list
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Only store every nth frame to match the desired fps
            if idx % frame_skip == 0:
                # Convert BGR to RGB format (OpenCV loads images in BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert OpenCV image to PIL image
                pil_img = Image.fromarray(frame_rgb)

                if max_dimension is not None and max(pil_img.size) > max_dimension:
                    # Resize the image to the max dimension
                    w, h = pil_img.size
                    if w > h:
                        new_w = max_dimension
                        new_h = int(h * (new_w / w))
                    else:
                        new_h = max_dimension
                        new_w = int(w * (new_h / h))
                    pil_img = pil_img.resize((new_w, new_h))
                frames.append(pil_img)
        except ZeroDivisionError:
            print(f'ZeroDivisionError at frame {idx} (frame_skip = {frame_skip}, original_fps = {original_fps}, fps = {fps})')

        idx += 1

    cap.release()

    return frames


def save_frames_as_video(frames, fpath, fps=24):
    """
    Save a list of PIL Image frames as an mp4 video and then convert it to webm using FFmpeg.
    
    Parameters:
    - frames: List of PIL Image objects
    - fpath: Path where the mp4 video will be saved
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
    
    _, ext = os.path.splitext(fpath)
    if ext != '.webm':
        raise ValueError("Output file must be a webm video")

    temp_fpath = fpath.replace('.webm', '.mp4')
    
    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    out = cv2.VideoWriter(temp_fpath, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        # Convert PIL Image to NumPy array
        frame_np = np.array(frame)
        
        # Convert RGB to BGR format (as OpenCV works with BGR for writing videos)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        out.write(frame_bgr)
    
    # Release the VideoWriter
    out.release()

    print(f'Video saved to {temp_fpath}; converting to {fpath}')
    
    # Construct the FFmpeg command
    cmd = ['/usr/bin/ffmpeg', '-y', '-i', temp_fpath, '-c:v', 'libvpx-vp9', '-b:v', '1M', '-c:a', 'libopus', fpath]



    # Run the FFmpeg command
    subprocess.run(cmd, check=True)
    # remove mp4 temp file
    os.remove(temp_fpath)


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

from typing import List

def frames_to_np(frames: List[Image.Image]) -> np.ndarray:
    """
    Convert a list of PIL.Image frames to a 4D NumPy array.

    Parameters:
    frames (List[PIL.Image]): List of image frames

    Returns:
    np.ndarray: 4D NumPy array of shape (num_frames, height, width, 3)
    """
    # Initialize an empty list to store NumPy arrays
    array_list = []

    # Loop through each frame
    for frame in frames:
        # Convert PIL image to NumPy array and append to the list
        array_list.append(np.array(frame))

    # Stack arrays along a new axis
    return np.stack(array_list)
