import os
os.environ["HF_ENDPOINT"]          = "https://hf-mirror.com"
import argparse
from datetime import datetime
import logging
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import .src.anisoraV2_gpu.wan
from .src.anisoraV2_gpu.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from .src.anisoraV2_gpu.wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from .src.anisoraV2_gpu.wan.utils.utils import cache_video, cache_image, str2bool


class LoadAniSoraModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./Wan2.1-I2V-14B-480P"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "AniSora"

    def load_model(self, model_path):
        model = model_path
        return (model,)


class AniSoraPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "input_text"
    CATEGORY = "AniSora"

    def input_text(self, text):
        prompt = text
        return (prompt,)


class AniSora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "task": ("STRING", {"default": "t2v-14B"}),
                "ckpt_dir": ("MODEL",),
                "prompt": ("PROMPT",),
                "size": ("STRING", {"default": "1280*720"}),
                "frame_num": ("INT", {"default": 49}),
                "ulysses_size": ("INT", {"default": 1}),
                "ring_size": ("INT", {"default": 1}),
                "base_seed": ("INT", {"default": "4096"}),
                "sample_solver": ("STRING", {"default": ['unipc', 'dpm++']}),
                "sample_steps": ("INT", {"default": 50}),
                "sample_shift": ("FLOAT", {"default": 5}),
                "sample_guide_scale": ("FLOAT", {"default": 5.0}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "AniSora"

    def generate(self, task, size, frame_num, ckpt_dir, ulysses_size, ring_size, prompt, 
                base_seed, sample_solver, sample_steps, sample_shift, sample_guide_scale):

        offload_model = None
        t5_fsdp = False
        t5_cpu = False
        dit_fsdp = False
                  
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
    
        if offload_model is None:
            offload_model = False if world_size > 1 else True
            
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size)
        else:
            assert not (
                t5_fsdp or dit_fsdp
            ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            assert not (
                ulysses_size > 1 or ring_size > 1
            ), f"context parallel are not supported in non-distributed environments."
    
        if ulysses_size > 1 or ring_size > 1:
            assert ulysses_size * ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
            from xfuser.core.distributed import (initialize_model_parallel,
                                                 init_distributed_environment)
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())
    
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=ring_size,
                ulysses_degree=ulysses_size,
            )

        cfg = WAN_CONFIGS[task]
        if ulysses_size > 1:
            assert cfg.num_heads % ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."
    
        if dist.is_initialized():
            base_seed = [base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            base_seed = base_seed[0]
    
        if "t2v" in task or "t2i" in task:
            
            image=image
            prompt=prompt

            wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=dit_fsdp,
                use_usp=(ulysses_size > 1 or ring_size > 1),
                t5_cpu=t5_cpu,
            )

            video = wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[size],
                frame_num=frame_num,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=sample_guide_scale,
                seed=base_seed,
                offload_model=offload_model)
        
        return (video,)


class SaveAniSora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_path": ("STRING", {"default": "output"}),
                "video": ("VIDEO",),
                "task": ("STRING", {"default": "t2v-14B"}),
                "fps": ("INT", {"default": 16}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_model"
    CATEGORY = "AniSora"

    def save_model(self, save_path, video, task, fps):
      
        if "t2v" in task:
            save_file = save_path + '.mp4'
            cache_video(
                tensor=video[None],
                save_file=save_file,
                fps=fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
      
        return ()

