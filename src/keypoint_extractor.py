from PIL import Image
from typing import TypedDict
import torch 
from diffusers_extractor import SDFeatureExtractor
from extractors.dift_extractor import DIFTFeatureExtractor
from utils import pil_to_tensor

class Keypoint(TypedDict):
    x: int
    y: int
    frame: int
    id: str


class KeypointExtractor:

    def __init__(self, device='cuda', sd_id="stabilityai/stable-diffusion-2-1"):
        self.device = device
        self.feature_extractor = DIFTFeatureExtractor(sd_id, device=device)

    def img_to_features(self, images: list[Image.Image], prompt: str, verbose=False, layers=[1], steps=[261]) -> torch.Tensor:
        """
        Converts an image to a feature vector.
        Returns:
            torch.Tensor: feature vector
        """
        if type(images) != list:
            assert isinstance(images, Image.Image), f"KeypointExtractor.img_to_features(): images must be a list of PIL images or a single PIL image. Received type {type(images)}."
            images = [images]

        images_tensors = torch.stack([pil_to_tensor(img).to(self.device) for img in images])
        if verbose:
            print(f'KeypointExtractor.img_to_features(): images_tensors.shape = {images_tensors.shape}')
        
        # out is a tensor of shape [BS, CH, H, W] with features for each image
        out = extractor(images_tensors, layers=layers, prompt=prompt, steps=steps)

        return out


    def extract_keypoints(self, frames: list[Image.Image], source_frame: int = 0, grid_size = 30) -> list[Keypoint]:
        """
        Sets grid points on the source frame and tracks them for each frame in the video.
        Returns:
            list[Keypoint]: list of keypoint dictionaries for each frame in the video. Each keypoint dictionary has the following keys: x, y, frame, id
        """
        raise NotImplementedError("KeypointExtractor.extract_keypoints() is not implemented.")