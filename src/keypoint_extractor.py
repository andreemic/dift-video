from PIL import Image

import torch 
import torch.nn as nn
# from extractors.joha import SDFeatureExtractor
from extractors.dift_extractor import DIFTFeatureExtractor
from utils import pil_to_tensor, Keypoint, get_grid_keypoints
from performance import PerformanceManager, MockPerformanceManager

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import numpy as np

"""

class Keypoint(TypedDict):
    x: int
    y: int
    id: str
    idx: int

"""


class KeypointExtractor:

    def __init__(self, device='cuda', sd_id="stabilityai/stable-diffusion-2-1"):
        self.device = device
        self.feature_extractor = DIFTFeatureExtractor(sd_id, device=device)

    def img_to_features(self, images: list[Image.Image], prompt: str, verbose=False, layer=1, step=261, perf_manager=MockPerformanceManager) -> torch.Tensor:
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
        features = self.feature_extractor(images_tensors, prompt=prompt, perf_manager=perf_manager).squeeze(1).to("cpu")

        perf_manager.start('upsample_features')
        upsampling_to = (initial_h, initial_w)
        upsampled_features = nn.Upsample(size=upsampling_to, mode='bilinear')(features).to(self.device)
        perf_manager.end('upsample_features')

        return upsampled_features


    def track_keypoints(self, frames: list[Image.Image], prompt: str,  source_frame_idx: int=0, grid_size=30, source_frame_keypoints=None, perf_manager=MockPerformanceManager, parallel=False) -> list[list[Keypoint]]:
        """
        Tracks keypoints (or grid) on the source frame across each frame in the video.
        
        Args:
            frames (list[Image.Image]): list of PIL images
            source_frame_idx (int): index of the source frame in the frames list
            grid_size (int): size of the grid to use for keypoints
            source_frame_keypoints (list[Keypoint]): list of keypoints on the source frame. If None, a grid of keypoints will be generated.

        Returns:
            list[list[Keypoint]]: list of keypoint dictionaries for each frame in the video. Each keypoint dictionary has the following keys: x, y, frame, id
        """
        features = self.img_to_features(frames, prompt=prompt, perf_manager=perf_manager)
        if source_frame_keypoints is None:
            assert grid_size is not None, "grid_size must be specified if source_frame_keypoints is None"
            source_frame_keypoints = get_grid_keypoints(frames[source_frame_idx].size[0], frames[source_frame_idx].size[1], grid_size=grid_size)

        
        keypoints_per_frame = []

        for i, frame in tqdm(enumerate(frames), desc='Tracking keypoints', total=len(frames), unit='frame'):
            if i == source_frame_idx:
                keypoints_per_frame.append(source_frame_keypoints)
                continue
            else:
                perf_manager.start('get_frame_to_frame_correspondence')
                if parallel:
                    this_frame_keypoints = self.get_keypoints_correspondence_parallel(features[source_frame_idx], features[i], source_frame_keypoints)
                else:
                    this_frame_keypoints = self.get_keypoints_correspondence(features[source_frame_idx], features[i], source_frame_keypoints, perf_manager=perf_manager)
                perf_manager.end('get_frame_to_frame_correspondence')
                keypoints_per_frame.append(this_frame_keypoints)
        
        return keypoints_per_frame


    def get_keypoints_correspondence(self, source_feature, target_feature, source_frame_keypoints: list[Keypoint], perf_manager=MockPerformanceManager) -> list[Keypoint]:
        target_keypoints = []

        for source_keypoint in tqdm(source_frame_keypoints, desc="Finding correspondences for keypoints", total=len(source_frame_keypoints), unit='keypoint'):
            source_xy = (source_keypoint['x'], source_keypoint['y'])
            target_keypoints_xys = self.get_correspondence(source_feature, target_feature, source_xy, perf_manager=perf_manager)
            target_keypoints.append({
                # keep the id and idx from the source keypoint
                **source_keypoint,

                # add the x,y coordinates of the target keypoint in the target frame
                'x': target_keypoints_xys[0][0],
                'y': target_keypoints_xys[0][1],
            })
        return target_keypoints

    def get_keypoints_correspondence_parallel(self, source_feature, target_feature, source_frame_keypoints: list[Keypoint], perf_manager=MockPerformanceManager) -> list[Keypoint]:
        
        def task(source_keypoint):
            source_xy = (source_keypoint['x'], source_keypoint['y'])
            target_keypoints_xys = self.get_correspondence(source_feature, target_feature, source_xy, perf_manager=perf_manager)
            return {
                # keep the id and idx from the source keypoint
                **source_keypoint,

                # add the x,y coordinates of the target keypoint in the target frame
                'x': target_keypoints_xys[0][0],
                'y': target_keypoints_xys[0][1],
            }
        
        with ProcessPoolExecutor() as executor:
            target_keypoints = list(executor.map(task, source_frame_keypoints))
        
        return target_keypoints


    def get_correspondence(self, source_feature: torch.Tensor, target_feature: torch.Tensor, source_xy: tuple[int, int], perf_manager=MockPerformanceManager):
        """
        Returns the target x,y coordinates on the target frame that correspond to the source x,y coordinates on the source frame.

        Args:
            source_feature (torch.Tensor): feature map of the source frame. Shape: [1, CH, H, W]
            target_feature (torch.Tensor): feature map of the target frame. Shape: [N, CH, H, W]
            source_xy (tuple[int, int]): x,y coordinates of the source point on the source frame
        """
        if len(source_feature.shape) == 3:
            source_feature = source_feature.unsqueeze(0)
        if len(target_feature.shape) == 3:
            target_feature = target_feature.unsqueeze(0)
        
        
        perf_manager.start('get_correspondence:init_cos_similarity')
        cos = nn.CosineSimilarity(dim=1).to(self.device)
        perf_manager.end('get_correspondence:init_cos_similarity')

        x, y = source_xy
        num_channel = source_feature.size(1)

        perf_manager.start('get_correspondence:extract_source_vector')
        source_vector = source_feature[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1
        perf_manager.end('get_correspondence:extract_source_vector')
        
        perf_manager.start('get_correspondence:cos_similarity')
        cos_map = cos(source_vector, target_feature).cpu().numpy()  # N, H, W
        perf_manager.end('get_correspondence:cos_similarity')

        target_xys = []
        for i in range(len(cos_map)):
            perf_manager.start('get_correspondence:get_max_xy_from_cos_map')
            max_yx = np.unravel_index(cos_map[i].argmax(), cos_map[i].shape)
            perf_manager.end('get_correspondence:get_max_xy_from_cos_map')
            max_xy = (max_yx[1], max_yx[0])
            target_xys.append(max_xy)
        
        return target_xys
    