from argparse import Namespace
import os
import torch
import torch.nn as nn
from typing import Any, Dict
import warnings

from transformers import CLIPVisionConfig
from transformers import CLIPImageProcessor, SamImageProcessor
from PIL import Image
import numpy as np


class RADIOVisionTower(nn.Module):
    """
    Vision Tower for the RADIO model.

    Args:
        vision_tower (str): Vision tower name. This is passed on
            the command line with the `--vision_tower` argument.
            The string is expected in the pattern of:
            `radio:<image_size>:<checkpoint_or_version>:<extra_config>`.
            Where <extra_config> is a comma-separated list of key=value pairs.
            <image_size> is the image resolution.
            <checkpoint> is a TorchHub version or path to a checkpoint.
        args (Namespace): Arguments.
        delay_load (bool): Delay loading the model.
    """
    def __init__(self, vision_tower, args, delay_load=False):
        """Initialization Routine."""

        super().__init__()

        self.vision_tower_name = vision_tower[len("radio:"):]
        config_items = self.vision_tower_name.split(":")
        self.image_sizes = [int(x) for x in config_items[0].split(",")]
        if len(self.image_sizes) == 0:
            raise ValueError("Expected more than zero images sizes!")
        self.image_size = self.image_sizes[0]
        self.do_center_crop = args.mm_im_crop

        self.vision_tower_checkpoint = config_items[1]

        extra_config = {}
        if len(config_items) > 2:
            # Parse extra config items. These are provided as a comma-separated list
            # of key=value pairs.
            extra_config_items = config_items[2].split(",")

            for item in extra_config_items:
                key, value = item.split("=")
                extra_config[key] = value

        self.adaptor_name = extra_config.get("adaptor", "backbone")
        self.fuse_adaptor_with_backbone = eval(extra_config.get("fuse_adaptor_with_backbone", "False"))
        self.skip_layer_norm = eval(extra_config.get("skip_layer_norm", "False"))

        self.is_loaded = False

        if not delay_load:
            self.load_model()
        else:
            # FIXME: This is a hack to avoid having to load the config from the checkpoint.
            hidden_size = self.get_hidden_size()
            patch_size = 16

            self.cfg_only = CLIPVisionConfig(
                **{

                    "hidden_size": hidden_size,
                    "image_size": self.image_size,
                    "model_type": "radio_vision_model",
                    "num_attention_heads": None,
                    "num_channels": 3,
                    "num_hidden_layers": None,
                    "patch_size": patch_size,
                }
            )

    def get_hidden_size(self):
        if self.adaptor_name == "openai_clip":
            hidden_size = 1024
        elif self.adaptor_name == "clip":
            hidden_size = 1280
        elif self.adaptor_name == "rtx-translate":
            hidden_size = 2048
        elif self.adaptor_name == "backbone":
            hidden_size = 1280
        else:
            raise ValueError(f"Unknown adaptor name: {self.adaptor_name}")

        if self.fuse_adaptor_with_backbone:
            hidden_size += 1280

        return hidden_size

    @property
    def hidden_size(self):
        return self.get_hidden_size()

    def load_model(self):

        crop_size={'height': self.image_size, 'width': self.image_size}

        if self.do_center_crop:
            self.image_processor = CLIPImageProcessor(
                size={"shortest_edge": self.image_size},
                crop_size=crop_size,
                do_center_crop=self.do_center_crop,
                do_normalize=True,
            )
        else:
            self.image_processor = SamImageProcessor(
                    size={"longest_edge": self.image_size},
                    pad_size={'height': self.image_size, 'width': self.image_size},
                    do_pad=False,
                    do_normalize=True,
            )
            # Add a crop_size attribute to the image processor, since the
            # train.py script needs this to generate fake images of zeros
            # with the right size, when the sample does not have an
            # associated image.
            self.image_processor.crop_size = crop_size

        # For compatibility with CLIP Image Processor: the data loader uses width/height to
        # create dummy blank images for samples that don't have an image.
        self.image_processor.crop_size = {"width": self.image_size, "height": self.image_size}

        checkpoint_path_or_version = self.vision_tower_checkpoint

        # NOTE: do a lazy import of Timm to avoid issues with
        # DeepSpeed's ZeRO-3.
        from timm.models.vision_transformer import VisionTransformer

        self.vision_tower = torch.hub.load('NVlabs/RADIO',
                                           'radio_model',
                                           version=checkpoint_path_or_version,
                                           progress=True,
                                           adaptor_names=self.adaptor_name if self.adaptor_name != "backbone" else None)

        if isinstance(self.vision_tower.model, VisionTransformer):
            hidden_size = self.vision_tower.model.embed_dim
        else:
            raise ValueError(f"Unknown model type: {self.vision_tower}")

        # Override hidden size for OpenAI CLIP.
        hidden_size = self.get_hidden_size()

        if hasattr(self.vision_tower.model, "patch_generator"):
            patch_gen = self.vision_tower.model.patch_generator
            # Cropped Positional Embedding (CPE) case.
            patch_size = patch_gen.patch_size
        else:
            # Standard ViT case.
            patch_size = self.vision_tower.model.patch_embed.patch_size[0]

        self.vision_tower.config = CLIPVisionConfig(
                **{
                    "hidden_size": hidden_size,
                    "image_size": self.image_size,
                    "model_type": "radio_vision_model",
                    "num_attention_heads": None,
                    "num_channels": 3,
                    "num_hidden_layers": None,
                    "patch_size": patch_size,
                }
            )

        self.vision_tower.make_preprocessor_external()
        self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
        self._to_dtype = None

        if self.skip_layer_norm:
            self.vision_tower.model.norm = torch.nn.Identity()


    def to(self, *args, **kwargs):
        # Prevent casting the RADIO model's weights
        kwargs = dict(kwargs)
        self._to_dtype = kwargs.pop('dtype', None)
        super().to(*args, **kwargs)
        pass

    def train(self, mode=True):
        """Intercept call."""
        # Drop a warning if mode is True.
        if mode:
            warnings.warn("RADIOEncoder is always in eval mode.")
        pass

    @torch.no_grad()
    def get_features(self, x: torch.Tensor):
        output = self.vision_tower(x)
        if isinstance(output, dict):
            _, features = output[self.adaptor_name]
            if self.fuse_adaptor_with_backbone:
                _, backbone_features = output["backbone"]
                features = torch.cat([features, backbone_features], dim=2)
        else:
            _, features = output
        return features

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """Main forward pass."""
        input_shape = images.shape

        x = images

        # Add a batch dimension if necessary.
        if len(input_shape) == 3:
            x = x.unsqueeze(0)

        # Convert the input to the model's dtype (we assume
        # that the model only has one dtype for all parameters).
        param0 = next(self.vision_tower.parameters())
        x = x.to(dtype=param0.dtype, device=param0.device)

        patch_size = self.vision_tower.config.patch_size

        if self.do_center_crop:
            # Crop the input to a multiple of patch size.
            _, _, H, W = x.shape

            H = H - (H % patch_size)
            W = W - (W % patch_size)

            x = x[:, :, :H, :W]
        else:
            # Pad to nearest multiple of patch size
            _, _, H, W = x.shape
            H = H + (patch_size - (H % patch_size)) % patch_size
            W = W + (patch_size - (W % patch_size)) % patch_size
            x = nn.functional.pad(x, (0, W - x.shape[3], 0, H - x.shape[2]), mode="constant", value=0)

        features = self.get_features(x) # B, T, C

        B, _, H, W = x.shape
        _, _, C = features.shape

        # Remove the batch dimension if we added it.
        if len(input_shape) == 3:
            features = features.squeeze(0)

        # Cast back to the input's dtype.
        features = features.to(images.dtype)

        assert features.shape[-1] == self.get_hidden_size()

        return features