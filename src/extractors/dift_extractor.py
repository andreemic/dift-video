# requires path to contain dift/src directory from the original DIFT repo
from dift_sd import SDFeaturizer
import torch

# small wrapper around SDFeatureExtractor to split layers/steps config from the extractor function
class DIFTFeatureExtractor(SDFeaturizer):
    def __init__(self, sd_id="stabilityai/stable-diffusion-2-1", device='cuda', layer=1, step=261):
        self.layer = layer
        self.step = step

        super().__init__(device=device, sd_id=sd_id)
    
    @torch.no_grad()
    def __call__(self, img_tensor, prompt, layer=None, step=None):
        if layer is None:
            layer = self.layer
        if step is None:
            steps = self.step

        results = []
        for i in range(len(img_tensor)):
            results.append(super().__call__(img_tensor, prompt, up_ft_index=layer, t=step))
        
        return torch.stack(results)
