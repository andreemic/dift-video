from PIL import Image
from typing import TypedDict

class Keypoint(TypedDict):
    x: int
    y: int
    frame: int
    id: str


class KeypointExtractor:
    def extract_keypoints(self, frames: list[Image.Image], source_frame: int = 0, grid_size = 30) -> list[Keypoint]:
        """
        Sets grid points on the source frame and tracks them for each frame in the video.
        Returns:
            list[Keypoint]: list of keypoint dictionaries for each frame in the video. Each keypoint dictionary has the following keys: x, y, frame, id
        """
        raise NotImplementedError("KeypointExtractor.extract_keypoints() is not implemented.")