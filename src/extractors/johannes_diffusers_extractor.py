# code by Johannes Fischer @ CompVis
import gc
import torch
import einops
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline
from diffusers.models.unet_2d_blocks import UpBlock2D
from diffusers.models.unet_2d_blocks import CrossAttnUpBlock2D
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.utils.import_utils import is_torch_version


def new_forward_UpBlock2D(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None,
                          store_intermediates=False):
    intermediates = []
    for resnet in self.resnets:
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            if is_torch_version(">=", "1.11.0"):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                )
            else:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
        else:
            hidden_states = resnet(hidden_states, temb)
        # print("\tResNet:", tuple(hidden_states.shape))
        if store_intermediates:
            intermediates.append(hidden_states.detach())


    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)
            # print("\tUpsampler:", tuple(hidden_states.shape))
            if store_intermediates:
                intermediates.append(hidden_states.detach())
        
    return hidden_states, intermediates


def new_forward_CrossAttnUpBlock2D(
    self,
    hidden_states: torch.FloatTensor,
    res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
    temb: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    upsample_size: Optional[int] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    store_intermediates=False
):
    intermediates = []
    for resnet, attn in zip(self.resnets, self.attentions):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        else:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        # print("\tResNet:", tuple(hidden_states.shape))
        if store_intermediates:
            intermediates.append(hidden_states.detach())

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)
            # print("\tUpsampler:", tuple(hidden_states.shape))
            if store_intermediates:
                intermediates.append(hidden_states.detach())

    return hidden_states, intermediates


class CustomUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        up_fts = {}
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            # print("--------------------- Middle block:\n\t", sample.shape)
            if 0 in up_ft_indices:
                up_fts[0] = sample
        
        # 5. up
        layer_counter = 1
        for i, upsample_block in enumerate(self.up_blocks):
            # print("--------------------- upsample block", i)
            
            if i > np.max(up_ft_indices):
                break
            
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            # new version with more control
            if isinstance(upsample_block, CrossAttnUpBlock2D):
                sample, intermediates = new_forward_CrossAttnUpBlock2D(
                    self=upsample_block,
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    store_intermediates=True
                )
            elif isinstance(upsample_block, UpBlock2D):
                sample, intermediates = new_forward_UpBlock2D(
                    self=upsample_block,
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    store_intermediates=True
                )
            else:
                raise NotImplementedError(f"Missing forward for {type(upsample_block)}")
            
            for layer_ft in intermediates:
                if layer_counter in up_ft_indices:
                    up_fts[layer_counter] = layer_ft
                layer_counter += 1

        return up_fts


class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output_fts = self.unet(latents_noisy, 
                                    t,
                                    up_ft_indices,
                                    encoder_hidden_states=prompt_embeds,
                                    cross_attention_kwargs=cross_attention_kwargs)
        return unet_output_fts


class SDFeatureExtractor:
    def __init__(self, sd_id='stabilityai/stable-diffusion-2-1', device=None):
        unet = CustomUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        self.device = device if device is not None else "cuda"
        onestep_pipe = onestep_pipe.to(self.device)
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        self.pipe = onestep_pipe

    @torch.no_grad()
    def __call__(self, 
                img_tensor,
                prompt, 
                layers=[5],
                steps=[101],
        ):
        '''
        Args:
            img_tensor: should be a torch tensor in the shape of [bs, c, h, w]
            prompt: the prompt to use, a string or a list of strings (length must
                match the batch-size of img_tensor)
            steps: the time steps to use, should be an list of ints in the 
                range of [0, 1000]
            layers: which upsampling layers of the U-Net to extract features
                from. With input (1, 3, 512, 512) and SD1.5 you can choose
                ---- bottleneck
                middle block: (1, 1280, 8, 8)   # 0
                ---- upsample block 0
                ResNet: (1, 1280, 8, 8)         # 1
                ResNet: (1, 1280, 8, 8)         # 2
                ResNet: (1, 1280, 8, 8)         # 3
                Upsampler: (1, 1280, 16, 16)    # 4
                ---- upsample block 1
                ResNet: (1, 1280, 16, 16)       # 5
                ResNet: (1, 1280, 16, 16)       # 6
                ResNet: (1, 1280, 16, 16)       # 7
                Upsampler: (1, 1280, 32, 32)    # 8
                ---- upsample block 2
                ResNet: (1, 640, 32, 32)        # 9
                ResNet: (1, 640, 32, 32)        # 10
                ResNet: (1, 640, 32, 32)        # 11
                Upsampler: (1, 640, 64, 64)     # 12
                ---- upsample block 3
                ResNet: (1, 320, 64, 64)        # 13
                ResNet: (1, 320, 64, 64)        # 14
                ResNet: (1, 320, 64, 64)        # 15
        Return:
            unet_fts: a two-level dictionary with keys being timesteps, values again are
                dictionaries with keys being the layer number and values being the
                respective timestep-layer feature map of shape (bs, c, h, w). e.g.:
                {101: {
                    2: (bs, 1280, 8, 8),
                    9: (bs, 640, 32, 32)
                },
                201: {...}
                }
        '''
        prompt_embeds = self.pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False) # [1, 77, dim]
        
        # check prompt
        if img_tensor.shape[0] != prompt_embeds.shape[0]:
            # if a prompt is given and batch-size != len(prompts) -> error
            # if empty prompt is given and batch-size != len(prompts) -> repeat embedding
            if isinstance(prompt, str):
                if prompt != "":
                    raise ValueError("Batch-size does not match number of prompts")
                else:
                    prompt_embeds = einops.repeat(prompt_embeds, '1 ... -> b ...', b=img_tensor.shape[0])
            # if list of prompts is given and list length does not match batch-size,
            # then definitely something went wrong!
            else:
                raise ValueError("Batch-size does not match number of prompts")
            
        img_tensor = img_tensor.to(self.device)
        out = {}
        for t in steps:
            unet_ft_all = self.pipe(
                img_tensor=img_tensor,
                t=t,
                up_ft_indices=layers,
                prompt_embeds=prompt_embeds)
            out[t] = unet_ft_all
        return out


if __name__ == "__main__":
    # sd_id = "runwayml/stable-diffusion-v1-5"
    sd_id = "stabilityai/stable-diffusion-2-1"
    device = "cuda:0"
    extractor = SDFeatureExtractor(sd_id=sd_id, device=device)

    # settings
    layers = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    steps = [101, 261]

    ipt = torch.randn((2, 3, 512, 512)).to(device)
    print("Input:", ipt.shape)
    out = extractor(ipt, layers=layers, prompt="", steps=steps)
    for tstep, layer_fts in out.items():
        print(f"---- Timestep: {tstep}")
        for layer, ft in layer_fts.items():
            print(f"\tlayer {layer}: {ft.shape}")


# small wrapper around SDFeatureExtractor to split layers/steps config from the extractor function
class FeatureExtractor(SDFeatureExtractor):
    def __init__(self, device='cuda', sd_id="stabilityai/stable-diffusion-2-1", layers=[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15], steps=[101, 261]):
        self.layers = layers
        self.steps = steps

        super().__init__(device=device, sd_id=sd_id)
    
    @torch.no_grad()
    def __call__(self, img_tensor, prompt, layers=None, steps=None):
        if layers is None:
            layers = self.layers
        if steps is None:
            steps = self.steps
        return super().__call__(img_tensor, prompt, layers=layers, steps=steps)
