import os
import numpy as np
import torch
import concurrent.futures

class FeatureCache():
    def __init__(self, device: str = 'cuda:0'):
        self.device = device

    def save_single_feature(self, save_data: tuple):
        """
        Save a single feature file.

        Args:
            save_data (tuple): Tuple containing (path, feature).
        """
        path, feature = save_data
        try:
            np.savez_compressed(path, feature.cpu().numpy())
        except Exception as e:
            print(f'⚠️ Failed to save feature to {path}: {e}')

    def save_features(self, features: torch.Tensor, folder_path: str) -> str:
        """
        Saves frame features to {index}.npz files in the specified folder.
        Args:
            features (`torch.Tensor`): tensor of shape [frame_num, ...]
            folder_path (`str`): path to the folder to save features to

        Returns:
            `str`: path to the folder with saved features
        """
        try:
            os.makedirs(folder_path, exist_ok=True)
            paths = [os.path.join(folder_path, f'{i}.npz') for i in range(len(features))]
            save_data = zip(paths, features)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(executor.map(self.save_single_feature, save_data))

            return folder_path
        except Exception as e:
            print(f'⚠️ Failed to save features to {folder_path}: {e}')
            return None

    def load_single_feature(self, feature_path: str) -> torch.Tensor:
        """
        Load a single feature file.

        Args:
            feature_path (str): The path to the .npz file.
        
        Returns:
            torch.Tensor: Loaded tensor.
        """
        return torch.from_numpy(np.load(feature_path)['arr_0']).to(self.device)

    def load_features(self, folder_path: str) -> torch.Tensor:
        """
        Loads frame features from {index}.npz files in the specified folder. 

        Args:
            folder_path (`str`): path to the folder to load features from
        
        Returns:
            torch.Tensor: tensor of shape [frame_num, ...] or None if no features were found
        """
        # Generate paths
        i = 0
        paths = []
        while True:
            feature_path = os.path.join(folder_path, f'{i}.npz')
            if os.path.exists(feature_path):
                paths.append(feature_path)
                i += 1
            else:
                break
        
        if len(paths) == 0:
            return None
        
        # Load features in parallel
        features = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for feature_tensor in executor.map(self.load_single_feature, paths):
                features.append(feature_tensor)
        
        return torch.stack(features)