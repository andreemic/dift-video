from PIL import Image
from typing import TypedDict
import torch 
import torch.nn as nn
# from extractors.joha import SDFeatureExtractor
from extractors.dift_extractor import DIFTFeatureExtractor
from utils import pil_to_tensor

import numpy as np
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
            torch.Tensor: feature vector of shape [BS, CH, H, W] (BS is 1 if a single image is passed); H and W are smaller than the original image size by a factor dependent on layer setting
        """
        if type(images) != list:
            assert isinstance(images, Image.Image), f"Images must be a list of PIL images or a single PIL image. Received type {type(images)}."
            images = [images]

        initial_w, initial_h = images[0].size
        assert all([img.size == (initial_w, initial_h) for img in images]), f"All images must have the same size. Received sizes {[img.size for img in images]}."

        # convert images to a big tensor of shape [BS, CH, H, W]
        images_tensors = torch.stack([pil_to_tensor(img).to(self.device) for img in images])
        if verbose:
            print(f'KeypointExtractor.img_to_features(): images_tensors.shape = {images_tensors.shape}')
        
        # out is a tensor of shape [BS, CH, H, W] with features for each image
        features = self.feature_extractor(images_tensors, prompt=prompt).squeeze(1)

        upsampling_to = (initial_h, initial_w)
        upsampled_features = nn.Upsample(size=upsampling_to, mode='bilinear')(features)

        return upsampled_features


    def extract_keypoints(self, frames: list[Image.Image], prompt: str,  source_frame: int = 0, grid_size = 30) -> list[Keypoint]:
        """
        Sets grid points on the source frame and tracks them for each frame in the video.
        Returns:
            list[Keypoint]: list of keypoint dictionaries for each frame in the video. Each keypoint dictionary has the following keys: x, y, frame, id
        """
        features = self.img_to_features(frames, prompt=prompt)

        
    def get_correspondence(self, source_feature: torch.Tensor, target_feature: torch.Tensor, source_xy: tuple[int, int]):
        """
        Returns the target x,y coordinates on the target frame that correspond to the source x,y coordinates on the source frame.

        Args:
            source_feature (torch.Tensor): feature map of the source frame. Shape: [1, CH, H, W]
            target_feature (torch.Tensor): feature map of the target frame. Shape: [N, CH, H, W]
            source_xy (tuple[int, int]): x,y coordinates of the source point on the source frame
        """
        cos = nn.CosineSimilarity(dim=1)
        x, y = source_xy
        num_channel = source_feature.size(1)

        source_vector = source_feature[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1
        
        cos_map = cos(source_vector, target_feature).cpu().numpy()  # N, H, W

        target_xys = []
        for i in range(len(cos_map)):
            max_yx = np.unravel_index(cos_map[i].argmax(), cos_map[i].shape)
            max_xy = (max_yx[1], max_yx[0])
            target_xys.append(max_xy)
        
        return target_xys
