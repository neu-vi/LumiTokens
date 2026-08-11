# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
# Modified for the LumiTokens public inference release, 2026.

import os
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
from lumitokens.utils import camera as camera_utils
from lumitokens.utils import data as data_utils
from .transformer import QK_Norm_TransformerBlock, init_weights
from .decoder_heads import build_decoder_head


class LatentSceneEditor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()

        # Initialize transformer blocks
        self._init_transformer()

        # Public inference does not need training losses or perceptual weights.
        if self.config.training.get("build_loss", True):
            from .loss import LossComputer

            self.loss_computer = LossComputer(config)
        else:
            self.loss_computer = nn.Identity()

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)

        return tokenizer

    def _create_linear_tokenizer(self, in_channels, d_model):
        """Tokenizer for token-wise features [B, T, C] -> [B, T, d_model]."""
        tokenizer = nn.Sequential(
            nn.Linear(in_channels, d_model, bias=False),
        )
        tokenizer.apply(init_weights)
        return tokenizer

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.image_tokenizer.in_channels,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )

        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.target_pose_tokenizer.in_channels,
            patch_size = self.config.model.target_pose_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )

        # Resolve which lighting tokenizers to create based on relight_signals
        signal_list = self.config.training.get("relight_signals", ["envmap"])
        if isinstance(signal_list, str):
            signal_list = [signal_list]
        use_envmap = "envmap" in signal_list
        use_point_light = "point_light" in signal_list
        if not use_envmap and not use_point_light:
            use_envmap = True  # fallback

        # Environment tokenizer (RGB + ray directions) - only if envmap in relight_signals
        if use_envmap:
            env_config = self.config.model.get("env_tokenizer", {})
            if env_config:
                self.env_tokenizer = self._create_tokenizer(
                    in_channels=env_config.get("in_channels", 9),
                    patch_size=env_config.get("patch_size", 8),
                    d_model=self.config.model.transformer.d
                )
            else:
                self.env_tokenizer = self._create_tokenizer(
                    in_channels=6,
                    patch_size=8,
                    d_model=self.config.model.transformer.d
                )
            print(f"Initialized env_tokenizer (relight_signals includes envmap)")
        else:
            self.env_tokenizer = None

        # Point-light tokenizer - only if point_light in relight_signals
        if use_point_light:
            pl_config = self.config.model.get("point_light_tokenizer", {})
            self.point_light_tokenizer = self._create_linear_tokenizer(
                in_channels=pl_config.get("in_channels", 10),
                d_model=self.config.model.transformer.d,
            )
            print(f"Initialized point_light_tokenizer (relight_signals includes point_light)")
        else:
            self.point_light_tokenizer = None

        # Decoder head (legacy MLP or multi-layer DPT).
        head_cfg = self.config.model.get("decoder_head", {})
        self.image_token_decoder, self.decoder_head_type = build_decoder_head(
            head_cfg=head_cfg,
            d_model=self.config.model.transformer.d,
            patch_size=self.config.model.target_pose_tokenizer.patch_size,
            out_channels=3,
        )
        dpt_cfg = head_cfg.get("dpt", {})
        self.decoder_head_layer_indices = dpt_cfg.get("layer_indices", [3, 6, 9, 12])


    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)

        # latent vectors for LVSM encoder-decoder
        self.n_light_field_latent = nn.Parameter(
            torch.randn(
                config.n_latent_vectors,
                config.d,
            )
        )
        nn.init.trunc_normal_(self.n_light_field_latent, std=0.02)

        # Create transformer blocks
        self.transformer_encoder = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.encoder_n_layer)
        ]

        self.transformer_decoder = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.decoder_n_layer)
        ]

        # Initialize albedo decoder if configured
        self.use_albedo_decoder = config.get("use_albedo_decoder", False)
        print(f"use_albedo_decoder: {self.use_albedo_decoder}")
        if self.use_albedo_decoder:
            self.transformer_decoder_albedo = [
                QK_Norm_TransformerBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm
                ) for _ in range(config.decoder_n_layer)
            ]

        # Apply special initialization if configured
        if config.get("special_init", False):
            # Encoder
            for idx, block in enumerate(self.transformer_encoder):
                if config.get("depth_init", False):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.encoder_n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))

            # Decoder
            for idx, block in enumerate(self.transformer_decoder):
                if config.get("depth_init", False):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.decoder_n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))

            # Albedo decoder (if enabled)
            if self.use_albedo_decoder:
                for idx, block in enumerate(self.transformer_decoder_albedo):
                    if config.get("depth_init", False):
                        weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    else:
                        weight_init_std = 0.02 / (2 * config.decoder_n_layer) ** 0.5
                    block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            # Encoder
            for block in self.transformer_encoder:
                block.apply(init_weights)

            # Decoder
            for block in self.transformer_decoder:
                block.apply(init_weights)

            # Albedo decoder (if enabled)
            if self.use_albedo_decoder:
                for block in self.transformer_decoder_albedo:
                    block.apply(init_weights)


        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        self.transformer_decoder = nn.ModuleList(self.transformer_decoder)
        if self.use_albedo_decoder:
            self.transformer_decoder_albedo = nn.ModuleList(self.transformer_decoder_albedo)
        self.transformer_input_layernorm_decoder = nn.LayerNorm(config.d, bias=False)

        # Transformer editor block (for editing latent tokens)
        editor_config = config.get("editor", {})
        editor_n_layer = editor_config.get("n_layer", 1)
        self.transformer_editor = nn.ModuleList([
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(editor_n_layer)
        ])
        # Initialize editor blocks with smaller weights to start near identity
        # This prevents the editor from disrupting pre-trained latent tokens initially
        editor_init_scale = editor_config.get("init_scale", 0.1)  # Default: 0.1x of normal init
        if config.get("special_init", False):
            for idx, block in enumerate(self.transformer_editor):
                if config.get("depth_init", False):
                    # Use smaller initialization for editor (scaled by editor_init_scale)
                    weight_init_std = (0.02 / (2 * (idx + 1)) ** 0.5) * editor_init_scale
                else:
                    weight_init_std = (0.02 / (2 * editor_n_layer) ** 0.5) * editor_init_scale
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            # Use smaller initialization for editor (scaled by editor_init_scale)
            editor_std = 0.02 * editor_init_scale
            for block in self.transformer_editor:
                block.apply(lambda module: init_weights(module, editor_std))


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()
        return self

    def freeze_reconstructor_and_renderer(self):
        """
        Freeze all layers used in reconstructor and renderer,
        only keep editor layers (transformer_editor and lighting tokenizers) trainable.

        Reconstructor layers:
        - image_tokenizer
        - n_light_field_latent
        - transformer_encoder

        Renderer layers:
        - target_pose_tokenizer
        - transformer_input_layernorm_decoder
        - transformer_decoder
        - image_token_decoder

        Editor layers (kept trainable):
        - env_tokenizer
        - point_light_tokenizer
        - transformer_editor
        """
        # Freeze reconstructor layers
        print("**********Freezing reconstructor layers...**********")
        for param in self.image_tokenizer.parameters():
            param.requires_grad = False
        self.n_light_field_latent.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False

        # Freeze renderer layers
        for param in self.target_pose_tokenizer.parameters():
            param.requires_grad = False
        for param in self.transformer_input_layernorm_decoder.parameters():
            param.requires_grad = False
        for param in self.transformer_decoder.parameters():
            param.requires_grad = False
        for param in self.image_token_decoder.parameters():
            param.requires_grad = False

        # Freeze albedo decoder if it exists
        if self.use_albedo_decoder:
            for param in self.transformer_decoder_albedo.parameters():
                param.requires_grad = False

        # Keep editor layers trainable (only those that exist)
        trainable_parts = []
        if self.env_tokenizer is not None:
            for param in self.env_tokenizer.parameters():
                param.requires_grad = True
            trainable_parts.append("env_tokenizer")
        if self.point_light_tokenizer is not None:
            for param in self.point_light_tokenizer.parameters():
                param.requires_grad = True
            trainable_parts.append("point_light_tokenizer")
        for param in self.transformer_editor.parameters():
            param.requires_grad = True
        trainable_parts.append("transformer_editor")
        print(f"Frozen reconstructor and renderer layers. Only editor layers ({', '.join(trainable_parts)}) are trainable.")

    def unfreeze_all(self):
        """Unfreeze all layers for full model training"""
        for param in self.parameters():
            param.requires_grad = True
        print("Unfrozen all layers.")

    def freeze_all_except_decoder_head(self):
        """Freeze all params except decoder head parameters."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.image_token_decoder.parameters():
            param.requires_grad = True
        print("Frozen all layers except decoder head.")

    def set_decoder_head_gate_schedule(self, global_step, gate_cfg):
        if self.decoder_head_type != "dpt":
            return
        if not hasattr(self.image_token_decoder, "set_gate_values"):
            return
        if gate_cfg is None:
            return

        gate_init = gate_cfg.get("gate_init", {"l3": 0.0, "l6": 0.0, "l9": 0.0, "l12": 1.0})
        gate_ramp_cfg = gate_cfg.get("gate_ramp_steps", {})
        ramp_len = int(gate_ramp_cfg.get("ramp_len", 1000))
        ramp_len = max(ramp_len, 1)

        gate_values = {
            "l3": float(gate_init.get("l3", 0.0)),
            "l6": float(gate_init.get("l6", 0.0)),
            "l9": float(gate_init.get("l9", 0.0)),
            "l12": float(gate_init.get("l12", 1.0)),
        }

        for key in ("l9", "l6", "l3"):
            start_step = int(gate_ramp_cfg.get(f"{key}_start", 0))
            if global_step <= start_step:
                continue
            progress = min(max((global_step - start_step) / ramp_len, 0.0), 1.0)
            gate_values[key] = gate_init.get(key, 0.0) + (1.0 - gate_init.get(key, 0.0)) * progress

        self.image_token_decoder.set_gate_values(gate_values)

    def set_dpt_transfer_stage(self, stage_name, transfer_cfg):
        stage_name = str(stage_name).lower()
        freeze_backbone = bool(transfer_cfg.get("freeze_backbone_in_stage1", True))
        stage2_unfreeze = transfer_cfg.get("stage2_unfreeze", "all")

        if stage_name == "stage1":
            if freeze_backbone:
                self.freeze_all_except_decoder_head()
            else:
                self.unfreeze_all()
            return

        # stage2 / default
        self.unfreeze_all()
        if stage2_unfreeze == "decoder_only":
            for param in self.parameters():
                param.requires_grad = False
            for module in [self.transformer_input_layernorm_decoder, self.transformer_decoder, self.image_token_decoder]:
                for param in module.parameters():
                    param.requires_grad = True
            if self.use_albedo_decoder:
                for param in self.transformer_decoder_albedo.parameters():
                    param.requires_grad = True
            print("Stage2 with decoder_only: only decoder stack and head are trainable.")



    def pass_layers(
        self,
        transformer_blocks,
        input_tokens,
        gradient_checkpoint=False,
        checkpoint_every=1,
        return_intermediate=False,
        capture_layers=None,
    ):
        """
        Helper function to pass input tokens through all transformer blocks with optional gradient checkpointing.

        Args:
            input_tokens: Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The input tokens to process through the transformer blocks.
            gradient_checkpoint: bool, default False
                Whether to use gradient checkpointing to save memory during training.
            checkpoint_every: int, default 1
                Number of transformer layers to group together for gradient checkpointing.
                Only used when gradient_checkpoint=True.

        Returns:
            Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The processed tokens after passing through all transformer blocks.
        """
        num_layers = len(transformer_blocks)

        if return_intermediate:
            capture_set = set(capture_layers or [])
            layer_outputs = {}
            for layer_idx, layer in enumerate(transformer_blocks, start=1):
                if gradient_checkpoint:
                    input_tokens = torch.utils.checkpoint.checkpoint(
                        layer,
                        input_tokens,
                        use_reentrant=False,
                    )
                else:
                    input_tokens = layer(input_tokens)
                if layer_idx in capture_set:
                    layer_outputs[layer_idx] = input_tokens
            return input_tokens, layer_outputs

        if not gradient_checkpoint:
            # Standard forward pass through all layers
            for layer in transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens

        # Gradient checkpointing enabled - process layers in groups
        def _process_layer_group(tokens, start_idx, end_idx):
            """Helper to process a group of consecutive layers."""
            for idx in range(start_idx, end_idx):
                tokens = transformer_blocks[idx](tokens)
            return tokens

        # Process layer groups with gradient checkpointing
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                start_idx,
                end_idx,
                use_reentrant=False
            )

        return input_tokens

    def _extract_image_tokens(self, decoder_tokens, n_patches, n_latent_vectors):
        image_tokens, _ = decoder_tokens.split([n_patches, n_latent_vectors], dim=1)
        return image_tokens

    def _decode_image_tokens(self, image_tokens, v_target, height, width, multi_scale_tokens=None):
        patch_size = self.config.model.target_pose_tokenizer.patch_size
        h_patches = height // patch_size
        w_patches = width // patch_size

        if self.decoder_head_type == "dpt":
            rendered_images = self.image_token_decoder(
                image_tokens=image_tokens,
                h_patches=h_patches,
                w_patches=w_patches,
                multi_scale_tokens=multi_scale_tokens,
            )
            rendered_images = rearrange(rendered_images, "(b v) c h w -> b v c h w", v=v_target)
            return rendered_images

        rendered_images = self.image_token_decoder(image_tokens)
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=h_patches,
            w=w_patches,
            p1=patch_size,
            p2=patch_size,
            c=3,
        )
        return rendered_images

    def _run_decoder_with_scales(
        self,
        decoder_tokens,
        n_patches,
        n_latent_vectors,
        checkpoint_every,
        gradient_checkpoint,
        use_albedo_decoder=False,
    ):
        decoder_blocks = self.transformer_decoder_albedo if use_albedo_decoder else self.transformer_decoder
        capture_layers = self.decoder_head_layer_indices if self.decoder_head_type == "dpt" else []
        if len(capture_layers) == 0:
            output_tokens = self.pass_layers(
                decoder_blocks,
                decoder_tokens,
                gradient_checkpoint=gradient_checkpoint,
                checkpoint_every=checkpoint_every,
            )
            image_tokens = self._extract_image_tokens(output_tokens, n_patches, n_latent_vectors)
            return image_tokens, None

        output_tokens, layer_outputs = self.pass_layers(
            decoder_blocks,
            decoder_tokens,
            gradient_checkpoint=gradient_checkpoint,
            checkpoint_every=checkpoint_every,
            return_intermediate=True,
            capture_layers=capture_layers,
        )
        image_tokens = self._extract_image_tokens(output_tokens, n_patches, n_latent_vectors)
        multi_scale_tokens = {}
        for layer_idx, layer_token in layer_outputs.items():
            multi_scale_tokens[layer_idx] = self._extract_image_tokens(layer_token, n_patches, n_latent_vectors)
        return image_tokens, multi_scale_tokens

    def decode_tokens_to_image(self, image_tokens, v_target, image_h_w, multi_scale_tokens=None):
        height, width = image_h_w
        return self._decode_image_tokens(
            image_tokens=image_tokens,
            v_target=v_target,
            height=height,
            width=width,
            multi_scale_tokens=multi_scale_tokens,
        )


    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        '''
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        '''

        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)

        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)

        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)

    def _point_rays_to_pose_cond(self, ray_o, ray_d, method="default_plucker"):
        """
        Args:
            ray_o: [b, v, n_rays, 3]
            ray_d: [b, v, n_rays, 3]
        Returns:
            pose_cond: [b, v, n_rays, 6] (default/custom) or [b, v, n_rays, 9] (aug)
        """
        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=-1, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=-1)
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=-1, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=-1)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=-1)
        else:
            o_cross_d = torch.cross(ray_o, ray_d, dim=-1)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=-1)
        return pose_cond

    def _build_env_tokens(self, input_dict, token_dim):
        if self.env_tokenizer is None:
            return None
        if not (hasattr(input_dict, 'env_ldr') and hasattr(input_dict, 'env_hdr') and hasattr(input_dict, 'env_dir')):
            return None
        if input_dict.env_ldr is None or input_dict.env_hdr is None or input_dict.env_dir is None:
            return None

        b, v_input = input_dict.env_dir.shape[:2]
        env_h, env_w = input_dict.env_dir.shape[3], input_dict.env_dir.shape[4]

        single_env_map = self.config.training.get("single_env_map", False)
        if single_env_map:
            if self.training:
                view_idx = torch.randint(0, v_input, (b,), device=input_dict.env_dir.device)
            else:
                view_idx = torch.zeros(b, dtype=torch.long, device=input_dict.env_dir.device)
            batch_indices = torch.arange(b, device=input_dict.env_dir.device)
            env_ldr = input_dict.env_ldr[batch_indices, view_idx].unsqueeze(1)
            env_hdr = input_dict.env_hdr[batch_indices, view_idx].unsqueeze(1)
            env_dir = input_dict.env_dir[batch_indices, view_idx].unsqueeze(1)
            v_input = 1
        else:
            env_ldr = input_dict.env_ldr
            env_hdr = input_dict.env_hdr
            env_dir = input_dict.env_dir

        if env_ldr.shape[3] != env_h or env_ldr.shape[4] != env_w:
            env_ldr = torch.nn.functional.interpolate(
                env_ldr.reshape(b * v_input, 3, env_ldr.shape[3], env_ldr.shape[4]),
                size=(env_h, env_w),
                mode='bilinear',
                align_corners=False
            ).reshape(b, v_input, 3, env_h, env_w)

        if env_hdr.shape[3] != env_h or env_hdr.shape[4] != env_w:
            env_hdr = torch.nn.functional.interpolate(
                env_hdr.reshape(b * v_input, 3, env_hdr.shape[3], env_hdr.shape[4]),
                size=(env_h, env_w),
                mode='bilinear',
                align_corners=False
            ).reshape(b, v_input, 3, env_h, env_w)

        directional_env = torch.cat([env_ldr, env_hdr, env_dir], dim=2)  # [b, v, 9, env_h, env_w]
        env_tokens = self.env_tokenizer(directional_env)  # [b*v, n_patch, d]
        _, n_env_patches, _ = env_tokens.size()
        env_tokens = env_tokens.reshape(b, v_input * n_env_patches, token_dim)
        return env_tokens

    def _build_point_light_tokens(self, input_dict, token_dim):
        if self.point_light_tokenizer is None:
            return None
        if not hasattr(input_dict, "point_light_rays"):
            return None
        if input_dict.point_light_rays is None:
            return None

        # [b, v, n_rays, 10] = intensity(1), color(3), ray_o(3), ray_d(3)
        point_rays = input_dict.point_light_rays
        b, v_input, n_rays, _ = point_rays.shape

        # Optional single-view point-light conditioning (similar to single_env_map)
        single_point_light = self.config.training.get("single_point_light", False)
        if single_point_light:
            if self.training:
                view_idx = torch.randint(0, v_input, (b,), device=point_rays.device)
            else:
                view_idx = torch.zeros(b, dtype=torch.long, device=point_rays.device)
            batch_indices = torch.arange(b, device=point_rays.device)
            point_rays = point_rays[batch_indices, view_idx].unsqueeze(1)  # [b,1,n,10]
            v_input = 1

        intensity_and_color = point_rays[..., :4]  # [b, v, n, 4]
        ray_o = point_rays[..., 4:7]
        ray_d = point_rays[..., 7:10]
        ray_d = ray_d / (torch.norm(ray_d, dim=-1, keepdim=True) + 1e-8)

        plucker_method = self.config.training.get("point_light_plucker_method", "default_plucker")
        ray_pose_cond = self._point_rays_to_pose_cond(ray_o, ray_d, method=plucker_method)
        point_light_features = torch.cat([intensity_and_color, ray_pose_cond], dim=-1)  # default: 4+6=10

        # Tokenize each ray as one token
        point_light_features = point_light_features.reshape(b, v_input * n_rays, point_light_features.shape[-1])
        point_tokens = self.point_light_tokenizer(point_light_features)  # [b, v*n_rays, d]
        return point_tokens

    def _build_editor_condition_tokens(self, input_dict, token_dim):
        # print('building conditional tokens...')
        signal_list = self.config.training.get("relight_signals", ["envmap"])
        if isinstance(signal_list, str):
            signal_list = [signal_list]
        # print("signal_list:", signal_list)

        cond_tokens = []
        if "envmap" in signal_list:
            env_tokens = self._build_env_tokens(input_dict, token_dim)
            if env_tokens is not None:
                cond_tokens.append(env_tokens)

        if "point_light" in signal_list:
            point_tokens = self._build_point_light_tokens(input_dict, token_dim)
            if point_tokens is not None:
                cond_tokens.append(point_tokens)

        if len(cond_tokens) == 0:
            return None
        if len(cond_tokens) == 1:
            return cond_tokens[0]
        return torch.cat(cond_tokens, dim=1)


    def reconstructor(self, input, checkpoint_every=None):
        """
        从输入图像重建场景的 latent_tokens

        Args:
            input: 包含 input.image, input.ray_o, input.ray_d 的 edict
            checkpoint_every: 梯度检查点间隔，如果为 None 则使用 config 中的值

        Returns:
            latent_tokens: [b, n_latent_vectors, d] - 编码后的场景潜在表示
            n_patches: int - 每个视图的 patch 数量
            d: int - token 维度
        """
        if checkpoint_every is None:
            checkpoint_every = self.config.training.grad_checkpoint_every

        #& Step 1: Build input image pose conditioning - Concatenate RGB images with ray info
        posed_input_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        b, v_input, c, h, w = posed_input_images.size()

        #& Step 2: Image tokenization - Split images into patches and convert to tokens
        input_img_tokens = self.image_tokenizer(posed_input_images)  # [b*v, n_patches, d]

        _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
        #*Reshape input tokens - Merge batch and view dimensions
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]

        #& Step 3: Initialize latent vectors - Learnable scene representation
        #* latent_vector_tokens: [b, 3072, 768] - Learnable scene latent tokens
        #* this corresponds to the scene latents in rayZer
        latent_vector_tokens = self.n_light_field_latent.expand(b, -1, -1) # [b, n_latent_vectors, d]
        encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1) # [b, n_latent_vectors + v*n_patches, d]

        #& Step 4: Transformer encoder - Process through 12 transformer blocks
        #* intermediate_tokens is the concatenation of the latent tokens and the input image tokens
        intermediate_tokens = self.pass_layers(
            self.transformer_encoder,
            encoder_input_tokens,
            gradient_checkpoint=self.training,
            checkpoint_every=checkpoint_every,
        )

        latent_tokens, input_img_tokens = intermediate_tokens.split(
            [self.config.model.transformer.n_latent_vectors, v_input * n_patches], dim=1
        ) # [b, n_latent_vectors, d], [b, v*n_patches, d]

        return latent_tokens, n_patches, d

    def renderer(self, latent_tokens, target, n_patches, d, checkpoint_every=None):
        """
        从场景 latent_tokens 和目标 ray maps 解码渲染结果

        Args:
            latent_tokens: [b, n_latent_vectors, d] - 场景潜在表示
            target: 包含 target.ray_o, target.ray_d, target.image_h_w 的 edict
            n_patches: int - 每个视图的 patch 数量
            d: int - token 维度
            checkpoint_every: 梯度检查点间隔，如果为 None 则使用 config 中的值

        Returns:
            rendered_images: [b, v_target, 3, h, w] - 渲染的 RGB 图像
        """
        if checkpoint_every is None:
            checkpoint_every = self.config.training.grad_checkpoint_every
        n_latent_vectors = self.config.model.transformer.n_latent_vectors

        #& Step 1: Build target pose conditioning - Only use pose info (no RGB image)
        target_pose_cond= self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()

        #& Step 2: Replicate latent tokens for each target view
        repeated_latent_tokens = repeat(
                                latent_tokens,
                                'b nl d -> (b v_target) nl d',
                                v_target=v_target)

        #& Step 3: Target pose tokenization - Convert target pose conditioning to tokens
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b*v_target, n_patches, d]

        #& Step 5: Put target pose tokens and latent tokens through decoder transformer
        decoder_input_tokens = torch.cat((target_pose_tokens, repeated_latent_tokens), dim=1) # [b*v_target, n_latent_vectors + n_patches, d]
        decoder_input_tokens = self.transformer_input_layernorm_decoder(decoder_input_tokens)

        target_image_tokens, multi_scale_tokens = self._run_decoder_with_scales(
            decoder_tokens=decoder_input_tokens,
            n_patches=n_patches,
            n_latent_vectors=n_latent_vectors,
            checkpoint_every=checkpoint_every,
            gradient_checkpoint=self.training,
            use_albedo_decoder=False,
        )

        height, width = target.image_h_w

        rendered_images = self._decode_image_tokens(
            image_tokens=target_image_tokens,
            v_target=v_target,
            height=height,
            width=width,
            multi_scale_tokens=multi_scale_tokens,
        )

        return rendered_images

    def renderer_albedo(self, latent_tokens, target, n_patches, d, checkpoint_every=None):
        """
        从场景 latent_tokens 和目标 ray maps 解码渲染 albedo 结果

        Args:
            latent_tokens: [b, n_latent_vectors, d] - 场景潜在表示
            target: 包含 target.ray_o, target.ray_d, target.image_h_w 的 edict
            n_patches: int - 每个视图的 patch 数量
            d: int - token 维度
            checkpoint_every: 梯度检查点间隔，如果为 None 则使用 config 中的值

        Returns:
            rendered_albedos: [b, v_target, 3, h, w] - 渲染的 albedo 图像
        """
        if checkpoint_every is None:
            checkpoint_every = self.config.training.grad_checkpoint_every
        n_latent_vectors = self.config.model.transformer.n_latent_vectors

        #& Step 1: Build target pose conditioning - Only use pose info (no RGB image)
        target_pose_cond= self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()

        #& Step 2: Replicate latent tokens for each target view
        repeated_latent_tokens = repeat(
                                latent_tokens,
                                'b nl d -> (b v_target) nl d',
                                v_target=v_target)

        #& Step 3: Target pose tokenization - Convert target pose conditioning to tokens
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b*v_target, n_patches, d]

        #& Step 4: Put target pose tokens and latent tokens through albedo decoder transformer
        decoder_input_tokens = torch.cat((target_pose_tokens, repeated_latent_tokens), dim=1) # [b*v_target, n_latent_vectors + n_patches, d]
        decoder_input_tokens = self.transformer_input_layernorm_decoder(decoder_input_tokens)

        target_albedo_tokens, multi_scale_tokens = self._run_decoder_with_scales(
            decoder_tokens=decoder_input_tokens,
            n_patches=n_patches,
            n_latent_vectors=n_latent_vectors,
            checkpoint_every=checkpoint_every,
            gradient_checkpoint=self.training,
            use_albedo_decoder=True,
        )

        height, width = target.image_h_w

        rendered_albedos = self._decode_image_tokens(
            image_tokens=target_albedo_tokens,
            v_target=v_target,
            height=height,
            width=width,
            multi_scale_tokens=multi_scale_tokens,
        )

        return rendered_albedos

    def _apply_editor_once(self, latent_tokens, condition_tokens):
        if condition_tokens is None:
            return latent_tokens
        editor_input_tokens = torch.cat([latent_tokens, condition_tokens], dim=1)
        checkpoint_every = self.config.training.grad_checkpoint_every
        editor_output_tokens = self.pass_layers(
            self.transformer_editor,
            editor_input_tokens,
            gradient_checkpoint=self.training,
            checkpoint_every=checkpoint_every
        )
        n_latent_vectors = self.config.model.transformer.n_latent_vectors
        return editor_output_tokens[:, :n_latent_vectors, :]

    def _run_multi_edit_chain(self, latent_tokens, input_dict, token_dim):
        chain_env_ldr = getattr(input_dict, "chain_env_ldr", None)
        chain_env_hdr = getattr(input_dict, "chain_env_hdr", None)
        if chain_env_ldr is None or chain_env_hdr is None:
            return latent_tokens, None

        if chain_env_ldr.dim() != 6 or chain_env_hdr.dim() != 6:
            return latent_tokens, None

        max_available_steps = min(chain_env_ldr.shape[1], chain_env_hdr.shape[1])
        if max_available_steps <= 0:
            return latent_tokens, None

        multi_edit_cfg = self.config.training.get("multi_edit", {})
        configured_max_steps = int(multi_edit_cfg.get("max_steps", 3))
        max_steps = max(1, min(configured_max_steps, max_available_steps))
        sample_mode = multi_edit_cfg.get("sample_mode", "uniform")
        force_all_steps = bool(multi_edit_cfg.get("force_all_steps", False))

        b = latent_tokens.shape[0]
        if force_all_steps:
            sampled_steps = torch.full((b,), max_steps, dtype=torch.long, device=latent_tokens.device)
        elif self.training:
            if sample_mode == "uniform":
                sampled_steps = torch.randint(1, max_steps + 1, (b,), device=latent_tokens.device)
            else:
                raise ValueError(f"Unsupported multi_edit.sample_mode: {sample_mode}")
        else:
            sampled_steps = torch.full((b,), max_steps, dtype=torch.long, device=latent_tokens.device)

        updated_latent_tokens = latent_tokens
        base_input = dict(input_dict)
        pass_latents = []
        for step_idx in range(max_steps):
            step_input = edict(base_input)
            step_input.env_ldr = chain_env_ldr[:, step_idx]
            step_input.env_hdr = chain_env_hdr[:, step_idx]
            condition_tokens = self._build_editor_condition_tokens(step_input, token_dim)
            edited_latent_tokens = self._apply_editor_once(updated_latent_tokens, condition_tokens)
            active_mask = (sampled_steps > step_idx).view(b, 1, 1)
            updated_latent_tokens = torch.where(active_mask, edited_latent_tokens, updated_latent_tokens)
            pass_latents.append(updated_latent_tokens)

        chain_info = {
            "sampled_steps": sampled_steps,
            "max_steps": max_steps,
            "sample_mode": sample_mode,
            "force_all_steps": force_all_steps,
            "pass_latents": torch.stack(pass_latents, dim=1),
        }
        return updated_latent_tokens, chain_info

    def _run_multi_forward_pass(self, input_dict, target_dict):
        chain_env_ldr = getattr(input_dict, "chain_env_ldr", None)
        chain_env_hdr = getattr(input_dict, "chain_env_hdr", None)
        if chain_env_ldr is None or chain_env_hdr is None:
            return None
        if chain_env_ldr.dim() != 6 or chain_env_hdr.dim() != 6:
            return None

        max_available_steps = min(chain_env_ldr.shape[1], chain_env_hdr.shape[1])
        if max_available_steps <= 0:
            return None

        multi_edit_cfg = self.config.training.get("multi_edit", {})
        configured_max_steps = int(multi_edit_cfg.get("max_steps", 3))
        max_steps = max(1, min(configured_max_steps, max_available_steps))
        sample_mode = multi_edit_cfg.get("sample_mode", "uniform")
        force_all_steps = bool(multi_edit_cfg.get("force_all_steps", False))

        b = input_dict.image.shape[0]
        if force_all_steps:
            sampled_steps = torch.full((b,), max_steps, dtype=torch.long, device=input_dict.image.device)
        elif self.training:
            if sample_mode != "uniform":
                raise ValueError(f"Unsupported multi_edit.sample_mode: {sample_mode}")
            sampled_steps = torch.randint(1, max_steps + 1, (b,), device=input_dict.image.device)
        else:
            sampled_steps = torch.full((b,), max_steps, dtype=torch.long, device=input_dict.image.device)

        context_query = edict(
            ray_o=input_dict.ray_o,
            ray_d=input_dict.ray_d,
            image_h_w=input_dict.image_h_w,
        )

        current_input_image = input_dict.image
        current_latent = None
        pass_latents = []
        pass_renders_target = []
        pass_renders_context = []
        n_patches = None
        d = None

        base_input = dict(input_dict)
        for step_idx in range(max_steps):
            step_input = edict(base_input)
            step_input.image = current_input_image
            step_input.env_ldr = chain_env_ldr[:, step_idx]
            step_input.env_hdr = chain_env_hdr[:, step_idx]

            latent_step, n_patches, d = self.reconstructor(step_input)
            condition_tokens = self._build_editor_condition_tokens(step_input, d)
            edited_latent_step = self._apply_editor_once(latent_step, condition_tokens)

            render_target_step = self.renderer(edited_latent_step, target_dict, n_patches, d)
            render_context_step = self.renderer(edited_latent_step, context_query, n_patches, d)

            active_mask_latent = (sampled_steps > step_idx).view(b, 1, 1)
            active_mask_image = (sampled_steps > step_idx).view(b, 1, 1, 1, 1)
            if current_latent is None:
                current_latent = edited_latent_step
                current_target_render = render_target_step
                current_context_render = render_context_step
            else:
                current_latent = torch.where(active_mask_latent, edited_latent_step, current_latent)
                current_target_render = torch.where(active_mask_image, render_target_step, current_target_render)
                current_context_render = torch.where(active_mask_image, render_context_step, current_context_render)

            pass_latents.append(current_latent)
            pass_renders_target.append(current_target_render)
            pass_renders_context.append(current_context_render)
            current_input_image = current_context_render

        return edict({
            "final_latent": current_latent,
            "final_render": current_target_render,
            "pass_latents": torch.stack(pass_latents, dim=1),
            "pass_renders_target": torch.stack(pass_renders_target, dim=1),
            "pass_renders_context": torch.stack(pass_renders_context, dim=1),
            "n_patches": n_patches,
            "d": d,
            "chain_info": edict({
                "sampled_steps": sampled_steps,
                "max_steps": max_steps,
                "sample_mode": sample_mode,
                "force_all_steps": force_all_steps,
            }),
        })

    def _select_chain_target_images(self, target_dict, sampled_steps, max_steps):
        chain_relit_images = getattr(target_dict, "chain_relit_images", None)
        if chain_relit_images is None or chain_relit_images.dim() != 6:
            return None
        if chain_relit_images.shape[1] < max_steps:
            max_steps = chain_relit_images.shape[1]
        if max_steps <= 0:
            return None

        selected_step = sampled_steps.clamp(min=1, max=max_steps) - 1
        gather_shape = [chain_relit_images.shape[0], 1] + list(chain_relit_images.shape[2:])
        gather_index = selected_step.view(-1, 1, 1, 1, 1, 1).expand(*gather_shape)
        selected_target = torch.gather(chain_relit_images, dim=1, index=gather_index).squeeze(1)
        return selected_target

    def _compute_multi_pass_outputs(self, chain_info, target, n_patches, d):
        pass_latents = chain_info["pass_latents"]  # [b, steps, n_latent, d]
        max_steps = pass_latents.shape[1]
        pass_renders = []
        for step_idx in range(max_steps):
            pass_renders.append(self.renderer(pass_latents[:, step_idx], target, n_patches, d))
        return torch.stack(pass_renders, dim=1)

    def _compute_multi_pass_loss(self, pass_renders, pass_targets, metric_prefix=""):
        # pass_renders/pass_targets: [b, steps, v, c, h, w]
        _, max_steps, _, _, _, _ = pass_renders.shape
        pass_metrics = []
        for pass_idx in range(max_steps):
            pass_metrics.append(
                self.loss_computer(
                    pass_renders[:, pass_idx],
                    pass_targets[:, pass_idx],
                )
            )

        loss_metrics = edict()
        metric_keys = list(pass_metrics[0].keys())
        for key in metric_keys:
            per_pass_vals = torch.stack([pm[key] for pm in pass_metrics], dim=0)
            key_name = f"{metric_prefix}{key}"
            loss_metrics[key_name] = per_pass_vals.mean()
            for pass_idx in range(max_steps):
                loss_metrics[f"pass{pass_idx + 1}_{key_name}"] = per_pass_vals[pass_idx]
        return loss_metrics

    def forward(self, data_batch, has_target_image=True, compute_loss=True):
        #& Step 1: Data preprocessing - Extract input and target data from data_batch, compute rays
        # input.image: [b, v_input, 3, h, w] - Input RGB images (v_input=2 views)
        # input.ray_o: [b, v_input, 3, h, w] - Ray origins
        # input.ray_d: [b, v_input, 3, h, w] - Ray directions
        # target.image: [b, v_target, 3, h, w] - Target RGB images (v_target=6 views)
        # target.ray_o: [b, v_target, 3, h, w] - Target ray origins
        # target.ray_d: [b, v_target, 3, h, w] - Target ray directions
        input, target = self.process_data(data_batch, has_target_image=has_target_image, target_has_input = self.config.training.target_has_input, compute_rays=True)

        #& Step 2/3/4: Reconstructor + Editor + Renderer
        chain_info = None
        pass_renders = None
        pass_targets = None
        pass_renders_context = None
        pass_targets_context = None

        # recon_only: skip the editor pass entirely (encoder -> decoder only) and
        # supervise against the same-scene target.image. Multi-edit is force-disabled.
        recon_only = bool(self.config.training.get("recon_only", False))

        multi_edit_cfg = self.config.training.get("multi_edit", {})
        multi_edit_enable = bool(multi_edit_cfg.get("enable", False)) and (not recon_only)
        multi_edit_mode = multi_edit_cfg.get("mode", "latent_chain")

        if multi_edit_enable and multi_edit_mode == "forward_chain":
            forward_chain_out = self._run_multi_forward_pass(input, target)
            if forward_chain_out is None:
                multi_edit_mode = "latent_chain"
            else:
                latent_tokens = forward_chain_out.final_latent
                rendered_images = forward_chain_out.final_render
                n_patches = forward_chain_out.n_patches
                d = forward_chain_out.d
                chain_info = forward_chain_out.chain_info
                pass_renders = forward_chain_out.pass_renders_target
                pass_renders_context = forward_chain_out.pass_renders_context

                chain_supervision = None
                if hasattr(target, "pass_relit_images") and target.pass_relit_images is not None:
                    chain_supervision = target.pass_relit_images
                elif hasattr(target, "chain_relit_images") and target.chain_relit_images is not None:
                    chain_supervision = target.chain_relit_images
                if chain_supervision is not None:
                    max_steps = min(chain_info.max_steps, chain_supervision.shape[1])
                    pass_renders = pass_renders[:, :max_steps]
                    pass_targets = chain_supervision[:, :max_steps]

                chain_supervision_context = None
                if hasattr(input, "pass_relit_images") and input.pass_relit_images is not None:
                    chain_supervision_context = input.pass_relit_images
                elif hasattr(input, "chain_relit_images") and input.chain_relit_images is not None:
                    chain_supervision_context = input.chain_relit_images
                if chain_supervision_context is not None:
                    max_steps_ctx = min(chain_info.max_steps, chain_supervision_context.shape[1])
                    pass_renders_context = pass_renders_context[:, :max_steps_ctx]
                    pass_targets_context = chain_supervision_context[:, :max_steps_ctx]

        if (not multi_edit_enable) or multi_edit_mode == "latent_chain":
            latent_tokens, n_patches, d = self.reconstructor(input)
            if multi_edit_enable:
                latent_tokens, chain_info = self._run_multi_edit_chain(latent_tokens, input, d)
            # recon_only: do NOT run the editor; render straight from reconstructor tokens.
            if (not recon_only) and ((not multi_edit_enable) or (chain_info is None)):
                condition_tokens = self._build_editor_condition_tokens(input, d)
                latent_tokens = self._apply_editor_once(latent_tokens, condition_tokens)

            rendered_images = self.renderer(latent_tokens, target, n_patches, d)
            if chain_info is not None:
                pass_renders = self._compute_multi_pass_outputs(chain_info, target, n_patches, d)
                chain_supervision = None
                if hasattr(target, "pass_relit_images") and target.pass_relit_images is not None:
                    chain_supervision = target.pass_relit_images
                elif hasattr(target, "chain_relit_images") and target.chain_relit_images is not None:
                    chain_supervision = target.chain_relit_images
                if chain_supervision is not None:
                    max_steps = min(chain_info["max_steps"], chain_supervision.shape[1])
                    pass_renders = pass_renders[:, :max_steps]
                    pass_targets = chain_supervision[:, :max_steps]

        #& Step 4.5: Process albedo decoder if enabled
        rendered_albedos = None
        if self.use_albedo_decoder:
            rendered_albedos = self.renderer_albedo(latent_tokens, target, n_patches, d)


        #& Step 5: Compute loss (if target images are provided)
        # loss_metrics contains: L2 loss, LPIPS loss, perceptual loss, etc.
        # Use relit_images target split if available, otherwise fall back to original image
        if has_target_image and compute_loss:
            chain_target_images = None
            if chain_info is not None:
                multi_edit_cfg = self.config.training.get("multi_edit", {})
                if not bool(multi_edit_cfg.get("final_only", True)):
                    raise ValueError("Only multi_edit.final_only=true is currently supported.")
                chain_target_images = self._select_chain_target_images(
                    target_dict=target,
                    sampled_steps=chain_info["sampled_steps"],
                    max_steps=chain_info["max_steps"],
                )

            # Check if relit_images are available in target (from relit scene).
            # recon_only always supervises against the same-scene target.image.
            if recon_only:
                target_images = target.image
            elif chain_target_images is not None:
                target_images = chain_target_images
            elif hasattr(target, 'relit_images') and target.relit_images is not None:
                target_images = target.relit_images
            else:
                target_images = target.image

            if pass_renders is not None and pass_targets is not None:
                if multi_edit_mode == "forward_chain":
                    target_loss_metrics = self._compute_multi_pass_loss(
                        pass_renders, pass_targets, metric_prefix="target_"
                    )
                    loss_metrics = edict(target_loss_metrics)
                    if pass_renders_context is not None and pass_targets_context is not None:
                        context_loss_metrics = self._compute_multi_pass_loss(
                            pass_renders_context, pass_targets_context, metric_prefix="context_"
                        )
                        for k, v in context_loss_metrics.items():
                            loss_metrics[k] = v
                        loss_metrics.loss = 0.5 * (
                            loss_metrics["target_loss"] + loss_metrics["context_loss"]
                        )
                    else:
                        loss_metrics.loss = loss_metrics["target_loss"]
                    loss_metrics.image_loss = loss_metrics.loss
                else:
                    loss_metrics = self._compute_multi_pass_loss(pass_renders, pass_targets)
            else:
                # Compute image loss
                image_loss_metrics = self.loss_computer(
                    rendered_images,
                    target_images
                )

                # Compute albedo loss if albedo decoder is enabled and target albedos are available
                if self.use_albedo_decoder and hasattr(target, 'albedos') and target.albedos is not None:
                    albedo_loss_metrics = self.loss_computer(
                        rendered_albedos,
                        target.albedos
                    )

                    # Combine losses: add image loss and albedo loss
                    # Sum all loss values
                    combined_loss = image_loss_metrics.loss + albedo_loss_metrics.loss

                    # Create combined loss metrics with both image and albedo losses
                    loss_metrics = edict(
                        loss=combined_loss,
                        image_loss=image_loss_metrics.loss,
                        albedo_loss=albedo_loss_metrics.loss,
                        l2_loss=image_loss_metrics.l2_loss + albedo_loss_metrics.l2_loss,
                        lpips_loss=image_loss_metrics.lpips_loss + albedo_loss_metrics.lpips_loss,
                        perceptual_loss=image_loss_metrics.perceptual_loss + albedo_loss_metrics.perceptual_loss,
                        psnr=image_loss_metrics.psnr,  # Keep image PSNR
                        norm_perceptual_loss=image_loss_metrics.norm_perceptual_loss + albedo_loss_metrics.norm_perceptual_loss,
                        norm_lpips_loss=image_loss_metrics.norm_lpips_loss + albedo_loss_metrics.norm_lpips_loss,
                    )
                else:
                    # If albedo decoder is enabled but no target albedos, still use image loss only
                    # (albedo decoder will still generate albedos, but no loss is computed)
                    loss_metrics = image_loss_metrics
            if chain_info is not None:
                sampled_steps = chain_info["sampled_steps"] if isinstance(chain_info, dict) else chain_info.sampled_steps
                max_steps = chain_info["max_steps"] if isinstance(chain_info, dict) else chain_info.max_steps
                loss_metrics.sampled_edit_steps_mean = sampled_steps.float().mean()
                for step in range(1, max_steps + 1):
                    step_ratio = (sampled_steps == step).float().mean()
                    loss_metrics[f"sampled_edit_step_{step}_ratio"] = step_ratio
        else:
            loss_metrics = None

        #& Step 5: Return results
        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images
            )

        if rendered_albedos is not None:
            result.render_albedo = rendered_albedos
        if pass_renders is not None:
            result.render_passes = pass_renders
        if pass_targets is not None:
            result.target_passes = pass_targets
        if pass_renders_context is not None:
            result.render_passes_context = pass_renders_context
        if pass_targets_context is not None:
            result.target_passes_context = pass_targets_context

        return result

    @torch.no_grad()
    def render_full_target_for_vis(self, data_batch, view_chunk_size=4):
        """
        Render ALL target frames in json order for visualization/video saving.

        Decoupled from the training forward: the training loss may operate on a
        capped/sampled target subset for memory safety, but for saved images we want
        the complete, ordered sequence. Rendering is chunked over target views to
        keep peak memory bounded, and uses the same renderer() path as training so
        the DPT multi-scale decode stays consistent.

        Returns an edict matching `forward`'s result structure (without loss):
            input, target (all frames, json order), render (all frames).
        """
        input, target = self.process_data(
            data_batch,
            has_target_image=True,
            target_has_input=self.config.training.target_has_input,
            compute_rays=True,
            force_full_target=True,
        )

        latent_tokens, n_patches, d = self.reconstructor(input)

        recon_only = bool(self.config.training.get("recon_only", False))
        multi_edit_cfg = self.config.training.get("multi_edit", {})
        multi_edit_enable = bool(multi_edit_cfg.get("enable", False)) and (not recon_only)
        multi_edit_mode = multi_edit_cfg.get("mode", "latent_chain")

        chain_info = None
        if multi_edit_enable and multi_edit_mode == "latent_chain":
            latent_tokens, chain_info = self._run_multi_edit_chain(latent_tokens, input, d)
        # recon_only: render directly from reconstructor tokens (no editor pass).
        if (not recon_only) and ((not multi_edit_enable) or (chain_info is None)):
            condition_tokens = self._build_editor_condition_tokens(input, d)
            latent_tokens = self._apply_editor_once(latent_tokens, condition_tokens)

        v_total = target.ray_o.shape[1]
        if view_chunk_size is None or view_chunk_size <= 0:
            view_chunk_size = v_total
        rendered_chunks = []
        for start in range(0, v_total, view_chunk_size):
            end = min(start + view_chunk_size, v_total)
            sub_target = edict(
                ray_o=target.ray_o[:, start:end],
                ray_d=target.ray_d[:, start:end],
                image_h_w=target.image_h_w,
            )
            rendered_chunks.append(
                self.renderer(latent_tokens, sub_target, n_patches, d, checkpoint_every=None)
            )
        rendered_images = torch.cat(rendered_chunks, dim=1)

        return edict(
            input=input,
            target=target,
            loss_metrics=None,
            render=rendered_images,
        )

    def edit_scene_with_env(self, latent_tokens, input_with_env):
        """
        Edit scene latent tokens using configured lighting conditions from input.

        Args:
            latent_tokens: [b, n_latent_vectors, d] - Scene latent representation
            input_with_env: Input dict containing env and/or point-light conditions

        Returns:
            edited_latent_tokens: [b, n_latent_vectors, d] - Edited scene latent representation
        """
        edited_latent_tokens = latent_tokens
        condition_tokens = self._build_editor_condition_tokens(input_with_env, latent_tokens.shape[-1])
        if condition_tokens is None:
            return edited_latent_tokens

        editor_input_tokens = torch.cat([latent_tokens, condition_tokens], dim=1)
        checkpoint_every = self.config.training.grad_checkpoint_every
        editor_output_tokens = self.pass_layers(
            self.transformer_editor,
            editor_input_tokens,
            gradient_checkpoint=True,
            checkpoint_every=checkpoint_every
        )
        n_latent_vectors = self.config.model.transformer.n_latent_vectors
        edited_latent_tokens = editor_output_tokens[:, :n_latent_vectors, :]
        return edited_latent_tokens


    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """
        Render a video from the model.

        Args:
            result: Edict from forward pass or just data
            traj_type: Type of trajectory
            num_frames: Number of frames to render
            loop_video: Whether to loop the video
            order_poses: Whether to order poses

        Returns:
            result: Updated with video rendering
        """

        if data_batch.input is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target

        # Prepare input tokens; [b, v, 3+6, h, w]
        posed_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        bs, v_input, c, h, w = posed_images.size()

        input_img_tokens = self.image_tokenizer(posed_images)  # [b*v_input, n_patches, d]
        _, n_patches, d = input_img_tokens.size()  # [b*v_input, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)  # [b, v_input*n_patches, d]

        latent_vector_tokens = self.n_light_field_latent.expand(bs, -1, -1) # [b, n_latent_vectors, d]
        encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1) # [b, n_latent_vectors + v*n_patches, d]

        # Process through encoder
        intermediate_tokens = self.pass_layers(self.transformer_encoder, encoder_input_tokens, gradient_checkpoint=False)
        latent_tokens, _ = intermediate_tokens.split(
            [self.config.model.transformer.n_latent_vectors, v_input * n_patches], dim=1
        ) # [b, n_latent_vectors, d]

        if traj_type == "interpolate":
            c2ws = input.c2w # [b, v, 4, 4]
            fxfycxcy = input.fxfycxcy #  [b, v, 4]
            device = input.c2w.device

            # Create intrinsics from fxfycxcy
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device) # [b, v, 3, 3]
            intrinsics[:, :,  0, 0] = fxfycxcy[:, :, 0]
            intrinsics[:, :,  1, 1] = fxfycxcy[:, :, 1]
            intrinsics[:, :,  0, 2] = fxfycxcy[:, :, 2]
            intrinsics[:, :,  1, 2] = fxfycxcy[:, :, 3]

            # Loop video if requested
            if loop_video:
                c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
                intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

            # Interpolate camera poses
            all_c2ws, all_intrinsics = [], []
            for b in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                    c2ws[b, :, :3, :4], intrinsics[b], num_frames, order_poses=order_poses
                )
                all_c2ws.append(cur_c2ws.to(device))
                all_intrinsics.append(cur_intrinsics.to(device))

            all_c2ws = torch.stack(all_c2ws, dim=0) # [b, num_frames, 3, 4]
            all_intrinsics = torch.stack(all_intrinsics, dim=0) # [b, num_frames, 3, 3]

            # Add homogeneous row to c2ws
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)

            # Convert intrinsics to fxfycxcy format
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]  # fx
            all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]  # fy
            all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]  # cx
            all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]  # cy

        # Compute rays for rendering
        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(
            fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device
        )

        # Get pose conditioning for target views
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o.to(input.image.device),
            ray_d=rendering_ray_d.to(input.image.device)
        )

        _, num_views, c, h, w = target_pose_cond.size()

        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [bs*v_target, n_patches, d]
        _, n_patches, d = target_pose_tokens.size()  # [b*v_target, n_patches, d]
        target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)  # [b, v_target*n_patches, d]

        view_chunk_size = 4
        video_rendering_list = []

        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)

            # Get current chunk of target pose tokens
            start_idx, end_idx = cur_chunk * n_patches, (cur_chunk + cur_view_chunk_size) * n_patches
            cur_target_pose_tokens = rearrange(target_pose_tokens[:, start_idx:end_idx,: ],
                                               "b (v_chunk p) d -> (b v_chunk) p d",
                                               v_chunk=cur_view_chunk_size, p=n_patches)

            cur_repeated_latent_tokens = repeat(
                latent_tokens,
                'b nl d -> (b v_chunk) nl d',
                v_chunk=cur_view_chunk_size
                )

            decoder_input_tokens = torch.cat((cur_target_pose_tokens, cur_repeated_latent_tokens), dim=1)
            decoder_input_tokens = self.transformer_input_layernorm_decoder(decoder_input_tokens)

            target_image_tokens, multi_scale_tokens = self._run_decoder_with_scales(
                decoder_tokens=decoder_input_tokens,
                n_patches=n_patches,
                n_latent_vectors=self.config.model.transformer.n_latent_vectors,
                checkpoint_every=1,
                gradient_checkpoint=False,
                use_albedo_decoder=False,
            )

            # Decode to images
            height, width = target.image_h_w
            video_rendering = self._decode_image_tokens(
                image_tokens=target_image_tokens,
                v_target=cur_view_chunk_size,
                height=height,
                width=width,
                multi_scale_tokens=multi_scale_tokens,
            ).cpu()

            video_rendering_list.append(video_rendering)

        # Combine all chunks
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering

        return data_batch

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            if os.path.exists(load_path) and load_path.endswith(".pt"):
                ckpt_paths = [load_path]
            else:
                ckpt_paths = []

        # Check if checkpoint files exist
        if len(ckpt_paths) == 0:
            print(f"No checkpoint files found in {load_path}")
            return None

        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}: {e}")
            return None

        checkpoint_state_dict = checkpoint["model"]

        # Check if transformer_decoder_albedo exists in checkpoint
        has_albedo_decoder_in_ckpt = any(
            key.startswith("transformer_decoder_albedo") for key in checkpoint_state_dict.keys()
        )

        # If we need albedo decoder but it doesn't exist in checkpoint, initialize from transformer_decoder
        if self.use_albedo_decoder and not has_albedo_decoder_in_ckpt:
            print("!!!transformer_decoder_albedo not found in checkpoint, initializing from transformer_decoder weights")
            # Copy weights from transformer_decoder to transformer_decoder_albedo
            # First collect all keys to copy (to avoid mutating dict during iteration)
            keys_to_copy = {}
            for i in range(len(self.transformer_decoder)):
                # Copy each transformer block
                src_prefix = f"transformer_decoder.{i}."
                tgt_prefix = f"transformer_decoder_albedo.{i}."

                for key, value in checkpoint_state_dict.items():
                    if key.startswith(src_prefix):
                        new_key = key.replace(src_prefix, tgt_prefix)
                        keys_to_copy[new_key] = value.clone()

            # Now add all collected keys to the state dict
            checkpoint_state_dict.update(keys_to_copy)

        incompatible = self.load_state_dict(checkpoint_state_dict, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.startswith("loss_computer.")]
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("loss_computer.")]
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint architecture mismatch. "
                f"Missing keys: {missing[:10]}; unexpected keys: {unexpected[:10]}"
            )
        return 0

    @torch.no_grad()
    def init_from_LVSM(self, lvsm_checkpoint_path):
        """
        从 Images2LatentScene 模型初始化 LatentSceneEditor

        Args:
            lvsm_checkpoint_path: Images2LatentScene 模型的 checkpoint 路径

        Returns:
            0 if successful, None if failed
        """
        # Import Images2LatentScene
        from .LVSM_scene_encoder_decoder import Images2LatentScene

        # Load checkpoint
        if os.path.isdir(lvsm_checkpoint_path):
            ckpt_names = [file_name for file_name in os.listdir(lvsm_checkpoint_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(lvsm_checkpoint_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [lvsm_checkpoint_path]

        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load LVSM checkpoint from {ckpt_paths[-1]}")
            return None

        # Get state dicts directly from checkpoint
        checkpoint_state_dict = checkpoint["model"]
        editor_state_dict = self.state_dict()

        # Copy matching weights from checkpoint to LatentSceneEditor
        matched_keys = []
        unmatched_keys = []
        ignored_keys = []

        # Handle albedo decoder weights from checkpoint
        # If we don't use albedo decoder but checkpoint has it, ignore those weights
        # If we use albedo decoder but checkpoint doesn't have it, we'll initialize from decoder later
        if not self.use_albedo_decoder:
            # Ignore albedo decoder weights if they exist in checkpoint
            for key in list(checkpoint_state_dict.keys()):
                if key.startswith("transformer_decoder_albedo"):
                    ignored_keys.append(key)
        else:
            # We use albedo decoder - check if checkpoint has it
            has_albedo_decoder_in_ckpt = any(
                key.startswith("transformer_decoder_albedo") for key in checkpoint_state_dict.keys()
            )

            if not has_albedo_decoder_in_ckpt:
                # Checkpoint doesn't have albedo decoder, initialize from transformer_decoder
                print("transformer_decoder_albedo not found in checkpoint, initializing from transformer_decoder weights")
                # Copy weights from transformer_decoder to transformer_decoder_albedo
                keys_to_copy = {}
                for i in range(len(self.transformer_decoder)):
                    # Copy each transformer block
                    src_prefix = f"transformer_decoder.{i}."
                    tgt_prefix = f"transformer_decoder_albedo.{i}."

                    for key, value in checkpoint_state_dict.items():
                        if key.startswith(src_prefix):
                            new_key = key.replace(src_prefix, tgt_prefix)
                            keys_to_copy[new_key] = value.clone()

                # Add copied keys to checkpoint_state_dict for matching
                checkpoint_state_dict.update(keys_to_copy)

        # Now match all keys
        for key in editor_state_dict.keys():
            if key in checkpoint_state_dict:
                # Check if shapes match
                if editor_state_dict[key].shape == checkpoint_state_dict[key].shape:
                    editor_state_dict[key] = checkpoint_state_dict[key]
                    matched_keys.append(key)
                else:
                    print(f"Warning: Shape mismatch for {key}: {editor_state_dict[key].shape} vs {checkpoint_state_dict[key].shape}")
                    unmatched_keys.append(key)
            else:
                unmatched_keys.append(key)

        # Optionally initialize editor from decoder weights
        editor_config = self.config.model.transformer.get("editor", {})
        init_editor_from_decoder = editor_config.get("init_from_decoder", False)

        if init_editor_from_decoder:
            # Copy weights from decoder to editor (useful since both process latent tokens)
            editor_n_layer = len(self.transformer_editor)
            decoder_n_layer = len(self.transformer_decoder)

            # Copy decoder weights to editor (use last N layers if editor has fewer layers)
            copy_start_idx = max(0, decoder_n_layer - editor_n_layer)
            copied_layers = 0

            for editor_idx in range(editor_n_layer):
                decoder_idx = copy_start_idx + editor_idx
                if decoder_idx < decoder_n_layer:
                    # Copy weights from decoder layer to editor layer
                    editor_block = self.transformer_editor[editor_idx]
                    decoder_block = self.transformer_decoder[decoder_idx]

                    editor_block.load_state_dict(decoder_block.state_dict(), strict=True)
                    copied_layers += 1

            print(f"  - Initialized {copied_layers}/{editor_n_layer} editor layers from decoder")

        # Load the combined state dict
        self.load_state_dict(editor_state_dict, strict=False)

        print(f"Initialized LatentSceneEditor from Images2LatentScene:")
        print(f"  - Matched keys: {len(matched_keys)}")
        print(f"  - Unmatched keys (using new weights): {len(unmatched_keys)}")
        if ignored_keys:
            print(f"  - Ignored keys (not used in this model): {len(ignored_keys)}")
        if unmatched_keys:
            print(f"  - New components: {', '.join(unmatched_keys)}")

        return 0
