# Adapted for the LumiTokens public inference release, 2026.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .transformer import init_weights


class MLPImageHead(nn.Sequential):
    def __init__(self, d_model, patch_size, out_channels=3, use_sigmoid=True):
        layers = [
            nn.LayerNorm(d_model, bias=False),
            nn.Linear(d_model, (patch_size ** 2) * out_channels, bias=False),
            nn.Sigmoid() if use_sigmoid else nn.Identity(),
        ]
        super().__init__(*layers)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.use_sigmoid = use_sigmoid
        self.apply(init_weights)


class DPTImageHead(nn.Module):
    """
    Multi-layer token fusion head.
    Expected token maps from decoder layers, e.g. {3, 6, 9, 12}.
    """

    def __init__(
        self,
        d_model,
        patch_size,
        out_channels=3,
        layer_indices=(3, 6, 9, 12),
        feature_dim=256,
        fusion_mode="add",
        upsample_mode="bilinear",
        out_activation="sigmoid",
        gate_init=None,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.layer_indices = [int(x) for x in layer_indices]
        self.feature_dim = int(feature_dim)
        self.fusion_mode = fusion_mode
        self.upsample_mode = upsample_mode
        self.out_activation = out_activation

        self.layer_projs = nn.ModuleDict(
            {
                str(layer): nn.Linear(d_model, self.feature_dim, bias=False)
                for layer in self.layer_indices
            }
        )
        self.layer_refines = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=False),
                    nn.GELU(),
                    nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=False),
                    nn.GELU(),
                )
                for layer in self.layer_indices
            }
        )

        self.fuse_refine = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

        self.to_patch_pixels = nn.Conv2d(
            self.feature_dim,
            self.out_channels * (self.patch_size ** 2),
            kernel_size=1,
            bias=False,
        )

        gate_init = gate_init or {}
        self.layer_gates = nn.ParameterDict(
            {
                str(layer): nn.Parameter(
                    torch.tensor(float(gate_init.get(f"l{layer}", gate_init.get(str(layer), 1.0))))
                )
                for layer in self.layer_indices
            }
        )

        self.apply(init_weights)

    def set_gate_values(self, gate_values):
        with torch.no_grad():
            for layer, value in gate_values.items():
                key = str(layer).replace("l", "")
                if key in self.layer_gates:
                    self.layer_gates[key].fill_(float(value))

    def get_gate_values(self):
        return {f"l{layer}": float(self.layer_gates[str(layer)].item()) for layer in self.layer_indices}

    def _resize(self, x, size):
        if tuple(x.shape[-2:]) == tuple(size):
            return x
        if self.upsample_mode in ("bilinear", "bicubic", "trilinear"):
            return F.interpolate(x, size=size, mode=self.upsample_mode, align_corners=False)
        return F.interpolate(x, size=size, mode=self.upsample_mode)

    def forward(self, image_tokens=None, h_patches=None, w_patches=None, multi_scale_tokens=None, **kwargs):
        if multi_scale_tokens is None or len(multi_scale_tokens) == 0:
            raise ValueError("DPTImageHead requires non-empty multi_scale_tokens.")
        if h_patches is None or w_patches is None:
            raise ValueError("DPTImageHead requires h_patches and w_patches.")

        stage_maps = {}
        for layer in self.layer_indices:
            if layer not in multi_scale_tokens:
                continue
            tokens = multi_scale_tokens[layer]  # [bv, n_patches, d]
            feat = self.layer_projs[str(layer)](tokens)
            feat = rearrange(feat, "bv (h w) c -> bv c h w", h=h_patches, w=w_patches)
            feat = self.layer_refines[str(layer)](feat)
            stage_maps[layer] = feat * self.layer_gates[str(layer)]

        if len(stage_maps) == 0:
            raise ValueError("None of configured layer_indices exist in multi_scale_tokens.")

        # Coarse -> fine fusion.
        ordered = sorted(stage_maps.keys(), reverse=True)
        fused = stage_maps[ordered[0]]
        for layer in ordered[1:]:
            target = stage_maps[layer]
            fused = self._resize(fused, target.shape[-2:])
            if self.fusion_mode == "concat":
                fused = torch.cat([fused, target], dim=1)
                fused = fused[:, : self.feature_dim, :, :] + fused[:, self.feature_dim :, :, :]
            else:
                fused = fused + target
            fused = self.fuse_refine(fused)

        fused = self._resize(fused, (h_patches, w_patches))
        patch_pixels = self.to_patch_pixels(fused)
        image = F.pixel_shuffle(patch_pixels, upscale_factor=self.patch_size)

        if self.out_activation == "sigmoid":
            image = torch.sigmoid(image)
        elif self.out_activation == "tanh":
            image = torch.tanh(image)
        return image


def build_decoder_head(head_cfg, d_model, patch_size, out_channels=3):
    head_type = "mlp"
    cfg = {}
    if head_cfg is not None:
        head_type = str(head_cfg.get("type", "mlp")).lower()
        cfg = head_cfg.get(head_type, {})

    if head_type == "dpt":
        return DPTImageHead(
            d_model=d_model,
            patch_size=patch_size,
            out_channels=out_channels,
            layer_indices=cfg.get("layer_indices", [3, 6, 9, 12]),
            feature_dim=cfg.get("feature_dim", 256),
            fusion_mode=cfg.get("fusion_mode", "add"),
            upsample_mode=cfg.get("upsample_mode", "bilinear"),
            out_activation=cfg.get("out_activation", "sigmoid"),
            gate_init=cfg.get("gate_init", {"l3": 0.0, "l6": 0.0, "l9": 0.0, "l12": 1.0}),
        ), head_type

    return MLPImageHead(
        d_model=d_model,
        patch_size=patch_size,
        out_channels=out_channels,
        use_sigmoid=cfg.get("use_sigmoid", True),
    ), "mlp"
