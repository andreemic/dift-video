import argparse
import os
from tqdm import tqdm
import logging
from keypoint_extractor import KeypointExtractor
from utils import load_frames
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_video_keypoints(frames_dir: str, out_dir: str, extractor_service):
    try:
        # List the frames
        frame_files = os.listdir(frames_dir)
        if not frame_files:
            logger.warning("No frame files found in the specified directory.")
            return

        # Create output directory if it does not exist
        os.makedirs(out_dir, exist_ok=True)
        frames, fpaths = load_frames(frames_dir)
        keypoint_dicts = extractor_service.extract_keypoints(frames, source_frame=0)
        for i, keypoint_dict in tqdm(enumerate(keypoint_dicts), total=len(keypoint_dicts), desc="Saving keypoints"):
            frame_fname = os.path.basename(fpaths[i])
            keypoint_json_fname = f"{frame_fname.split('.')[0]}.json"
            with open(os.path.join(out_dir, keypoint_json_fname), "w") as f:
                json.dump(keypoint_dict, f)
            
        


    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Extract keypoints from video frames. Saves a json file ({kpid: string, x: number, y: number}[]) for each frame in the specified output directory.")
    parser.add_argument("FRAMES_DIR", help="Directory containing video frames")
    parser.add_argument("OUT_DIR", help="Directory to save extracted keypoints")
    args = parser.parse_args()

    # Dependency injection: replace `KeypointExtractor()` with your actual implementation
    extractor_service = KeypointExtractor()

    extract_video_keypoints(args.FRAMES_DIR, args.OUT_DIR, extractor_service)

if __name__ == "__main__":
    main()
