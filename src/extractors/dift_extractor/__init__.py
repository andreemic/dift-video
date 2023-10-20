# requires path to contain dift/src directory from the original DIFT repo

import sys
import os
def absolutize_path(relative_path_to_this_file):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path_to_this_file))

sys.path.append(absolutize_path('dift/src'))


from .dift.src.models.dift_sd import SDFeaturizer
import torch
from tqdm import tqdm

# small wrapper around SDFeatureExtractor to split layers/steps config from the extractor function
class DIFTFeatureExtractor(SDFeaturizer):
    def __init__(self, sd_id="stabilityai/stable-diffusion-2-1", device='cuda', layer=1, step=261):
        self.layer = layer
        self.step = step

        super().__init__(device=device, sd_id=sd_id)
    
    @torch.no_grad()
    def __call__(self, img_tensor, prompt, layer=None, step=None, perf_manager=None):
        if layer is None:
            layer = self.layer
        if step is None:
            step = self.step

        results = []
        for i in tqdm(range(len(img_tensor)), desc='Extracting features from images'):
            if perf_manager:
                perf_manager.start('extract_features_from_image')
            results.append(super(DIFTFeatureExtractor, self).forward(img_tensor[i], prompt, up_ft_index=layer, t=step))
            if perf_manager:
                perf_manager.end('extract_features_from_image')
        
        return torch.stack(results)
