# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
# Modified for the LumiTokens public inference release, 2026.

import os
import random
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
import imageio
from PIL import Image



def create_video_from_frames(frames, output_video_file, framerate=30):
    """
    Creates a video from a sequence of frames.

    Parameters:
        frames (numpy.ndarray): Array of image frames (shape: N x H x W x C).
        output_video_file (str): Path to save the output video file.
        framerate (int, optional): Frames per second for the video. Default is 30.
    """
    frames = np.asarray(frames)

    # Normalize frames if values are in [0,1] range
    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(output_video_file, frames, fps=framerate, quality=8)


def tensor_chw01_to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    """[c,h,w] float in ~[0,1] -> uint8 [h,w,c] RGB."""
    x = img_chw.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def save_eval_triplet_layout(
    out_root: str,
    safe_scene_name: str,
    context_images: torch.Tensor,
    context_frame_indices,
    gt_images: torch.Tensor,
    target_frame_indices,
    pred_images: torch.Tensor,
):
    """
    Save evaluation images under:
        out_root/context/<safe_scene_name>/<frame_idx>.png
        out_root/gt/<safe_scene_name>/<frame_idx>.png
        out_root/predicted/<safe_scene_name>/<frame_idx>.png

    Context uses input (reference) view indices; gt/predicted use target view indices.
    Filenames are zero-padded original frame indices for alignment across folders.
    """
    ctx_dir = os.path.join(out_root, "context", safe_scene_name)
    gt_dir = os.path.join(out_root, "gt", safe_scene_name)
    pred_dir = os.path.join(out_root, "predicted", safe_scene_name)
    for d in (ctx_dir, gt_dir, pred_dir):
        os.makedirs(d, exist_ok=True)

    for vi, fi in enumerate(context_frame_indices):
        arr = tensor_chw01_to_uint8_hwc(context_images[vi])
        Image.fromarray(arr).save(os.path.join(ctx_dir, f"{int(fi):05d}.png"))

    n_t = len(target_frame_indices)
    for vi in range(n_t):
        fi = int(target_frame_indices[vi])
        gt_arr = tensor_chw01_to_uint8_hwc(gt_images[vi])
        pr_arr = tensor_chw01_to_uint8_hwc(pred_images[vi])
        Image.fromarray(gt_arr).save(os.path.join(gt_dir, f"{fi:05d}.png"))
        Image.fromarray(pr_arr).save(os.path.join(pred_dir, f"{fi:05d}.png"))


class ProcessData(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Keep dataset sampling behavior unchanged while allowing encoder-side
        # single-view conditioning when explicitly enabled.
        self.single_view_input_mode = bool(self.config.training.get("single_view_input_mode", False))

    @torch.no_grad()
    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, device="cuda"):
        """
        Args:
            c2w (torch.tensor): [b, v, 4, 4]
            fxfycxcy (torch.tensor): [b, v, 4]
            h (int): height of the image
            w (int): width of the image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """

        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)

        fx, fy, cx, cy = fxfycxcy[:,:, 0], fxfycxcy[:,:,  1], fxfycxcy[:,:,  2], fxfycxcy[:,:,  3]
        h_orig = int(2 * cy.max().item())  # Original height (estimated from the intrinsic matrix)
        w_orig = int(2 * cx.max().item())  # Original width (estimated from the intrinsic matrix)
        if h is None or w is None:
            h, w = h_orig, w_orig

        # in case the ray/image map has different resolution than the original image
        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(device), x.to(device)
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d

    @torch.no_grad()
    def compute_env_dir(self, c2w, envmap_h=256, envmap_w=512, device="cuda"):
        """
        Compute environment map viewing directions in world space.

        Args:
            c2w (torch.tensor): [b, v, 4, 4] camera-to-world transformation matrices (OpenCV convention)
            envmap_h (int): height of the environment map, default 256
            envmap_w (int): width of the environment map, default 512
            device (str): device to compute on, default "cuda"

        Returns:
            env_dir (torch.tensor): [b, v, 3, envmap_h, envmap_w] viewing directions in world space
        """
        b, v = c2w.size()[:2]

        # Generate environment map directions (similar to generate_envir_map_dir)
        lat_step_size = np.pi / envmap_h
        lng_step_size = 2 * np.pi / envmap_w
        theta, phi = torch.meshgrid([
            torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, envmap_h, device=device),
            torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, envmap_w, device=device)
        ], indexing='ij')

        # Compute view directions in local environment map coordinate system
        # [envmap_h, envmap_w, 3]
        view_dirs_local = torch.stack([
            torch.cos(phi) * torch.cos(theta),
            torch.sin(phi) * torch.cos(theta),
            torch.sin(theta)
        ], dim=-1)  # [envmap_h, envmap_w, 3]

        # Reshape to [envmap_h * envmap_w, 3]
        view_dirs_local = view_dirs_local.reshape(-1, 3)  # [envmap_h * envmap_w, 3]

        # Convert c2w to w2c rotation (OpenCV convention)
        # c2w is [b, v, 4, 4], we need w2c rotation which is c2w[:3, :3].T
        c2w_rotation = c2w[:, :, :3, :3]  # [b, v, 3, 3]
        w2c_rotation = c2w_rotation.transpose(-2, -1)  # [b, v, 3, 3]

        # Apply Blender convention transformation
        # We need to apply two transformations:
        # 1. OpenCV c2w -> Blender c2w: multiply by diag(1, -1, -1) on the right
        #    R_blender = R_opencv @ T_cv2b
        #    Then w2c_blender = R_blender.T = (R_opencv @ T_cv2b).T = T_cv2b.T @ R_opencv.T
        #    Since T_cv2b is diagonal, T_cv2b.T = T_cv2b
        # 2. Blender w2c -> Gaffer World: multiply by axis_aligned_transform on the left
        #    R_final = axis_aligned_transform @ w2c_blender
        #    R_final = axis_aligned_transform @ (T_cv2b @ w2c_opencv)
        #    R_final = (axis_aligned_transform @ T_cv2b) @ w2c_opencv

        # axis_aligned_transform (Gaffer): [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        # T_cv2b (Blender<->OpenCV):       [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        # Combined = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]

        axis_aligned_transform = torch.tensor([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=torch.float32, device=device)  # [3, 3]

        # Apply transformation: axis_aligned_R = axis_aligned_transform @ w2c_rotation
        # Expand axis_aligned_transform to match batch dimensions
        axis_aligned_transform_expanded = axis_aligned_transform.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        axis_aligned_R = torch.matmul(axis_aligned_transform_expanded, w2c_rotation)  # [b, v, 3, 3]
        # set the align_R to be identity,
        # Transform view directions to world space
        # view_dirs_local: [envmap_h * envmap_w, 3]
        # axis_aligned_R: [b, v, 3, 3]
        # We need to apply: view_dirs_world = view_dirs_local @ axis_aligned_R
        # Expand view_dirs_local to [b, v, envmap_h * envmap_w, 3]
        view_dirs_local_expanded = view_dirs_local.unsqueeze(0).unsqueeze(0).expand(b, v, -1, -1)  # [b, v, envmap_h * envmap_w, 3]
        # Matrix multiplication: [b, v, envmap_h * envmap_w, 3] @ [b, v, 3, 3] = [b, v, envmap_h * envmap_w, 3]
        view_dirs_world = torch.matmul(view_dirs_local_expanded, axis_aligned_R)  # [b, v, envmap_h * envmap_w, 3]

        # Reshape to [b, v, 3, envmap_h, envmap_w]
        env_dir = view_dirs_world.reshape(b, v, envmap_h, envmap_w, 3).permute(0, 1, 4, 2, 3)  # [b, v, 3, envmap_h, envmap_w]

        return env_dir

    def fetch_views(self, data_batch, has_target_image=False, target_has_input=True, force_full_target=False):
        """
        Splits the input data batch into input and target sets.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w], optional for some target views
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            target_has_input (bool): If True, target includes input views.
            force_full_target (bool): If True and load_all_frames is enabled, bypass the
                training target cap and return ALL frames in json order. Used for
                visualization/video saving so the saved sequence is complete and ordered.

        Returns:
            tuple: (input_dict, target_dict), both as EasyDict objects.

        """
        # randomize input views if dynamic_input_view_num is True and not in inference mode
        if (self.config.training.get("dynamic_input_view_num", False)
            and (not self.config.inference.get("if_inference", False))):
            self.config.training.num_input_views = np.random.randint(2, 5)


        input_dict, target_dict = {}, {}
        encoder_num_input_views = 1 if self.single_view_input_mode else int(self.config.training.num_input_views)

        # load_all mode: dataset has already reordered views as
        # [evenly-sampled condition views..., remaining views in original order].
        # Keep deterministic split for input. For training memory safety, target
        # can be capped to a sampled subset while preserving temporal order.
        load_all_frames = bool(self.config.training.get("load_all_frames", False))
        if load_all_frames:
            is_inference = bool(self.config.inference.get("if_inference", False))
            max_target_views = int(self.config.training.get("load_all_max_target_views", 32))
            num_input = encoder_num_input_views
            num_views = data_batch["c2w"].size(1)
            bs = data_batch["c2w"].size(0)
            target_positions = None

            if (not is_inference) and (not force_full_target) and max_target_views > 0 and num_views > max_target_views:
                device = data_batch["c2w"].device
                sampled_idx = []
                for _ in range(bs):
                    cur = torch.randperm(num_views, device=device)[:max_target_views]
                    sampled_idx.append(torch.sort(cur).values)
                target_positions = torch.stack(sampled_idx, dim=0)  # [b, cap]
            else:
                device = data_batch["c2w"].device
                target_positions = torch.arange(num_views, device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1)

            # Restore target temporal order using original frame ids in batch["index"][..., 0].
            # This keeps input ordering unchanged while making target sequence strictly monotonic.
            if "index" in data_batch and isinstance(data_batch["index"], torch.Tensor) and data_batch["index"].dim() >= 3:
                frame_ids = torch.gather(data_batch["index"][:, :, 0], dim=1, index=target_positions)
                frame_sort = torch.argsort(frame_ids, dim=1)
                target_positions = torch.gather(target_positions, dim=1, index=frame_sort)

            for key, value in data_batch.items():
                if not isinstance(value, torch.Tensor):
                    input_dict[key] = value
                    target_dict[key] = value
                    continue
                if value.dim() < 2:
                    input_dict[key] = value
                    target_dict[key] = value
                    continue
                if key.startswith("chain_") and value.dim() >= 3:
                    input_dict[key] = value[:, :, :num_input, ...]
                    to_expand_dim = value.shape[3:]
                    expanded_index = target_positions.view(
                        target_positions.shape[0], 1, target_positions.shape[1], *(1,) * len(to_expand_dim)
                    ).expand(-1, value.shape[1], -1, *to_expand_dim)
                    target_dict[key] = torch.gather(value, dim=2, index=expanded_index)
                    if key == "chain_relit_images":
                        input_dict["pass_relit_images"] = input_dict[key]
                        target_dict["pass_relit_images"] = target_dict[key]
                else:
                    input_dict[key] = value[:, :num_input, ...]
                    to_expand_dim = value.shape[2:]
                    expanded_index = target_positions.view(
                        target_positions.shape[0], target_positions.shape[1], *(1,) * len(to_expand_dim)
                    ).expand(-1, -1, *to_expand_dim)
                    target_dict[key] = torch.gather(value, dim=1, index=expanded_index)

            height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
            input_dict["image_h_w"] = (height, width)
            target_dict["image_h_w"] = (height, width)
            return edict(input_dict), edict(target_dict)

        # render_all_views mode: input = first num_input_views, target = ALL views
        render_all_views = self.config.inference.get("render_all_views", False)
        is_inference = self.config.inference.get("if_inference", False)
        if render_all_views and is_inference:
            num_input = encoder_num_input_views
            for key, value in data_batch.items():
                if not isinstance(value, torch.Tensor):
                    input_dict[key] = value
                    target_dict[key] = value
                    continue
                input_dict[key] = value[:, :num_input, ...]
                target_dict[key] = value  # ALL views as targets

            height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
            input_dict["image_h_w"] = (height, width)
            target_dict["image_h_w"] = (height, width)
            return edict(input_dict), edict(target_dict)

        # same_pose mode: clone input views to target (no novel-view sampling)
        same_pose = self.config.inference.get("same_pose", False)
        if same_pose:
            for key, value in data_batch.items():
                if not isinstance(value, torch.Tensor):
                    input_dict[key] = value
                    target_dict[key] = value
                    continue
                input_dict[key] = value[:, :encoder_num_input_views, ...]
                target_dict[key] = input_dict[key].clone()

            height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
            input_dict["image_h_w"] = (height, width)
            target_dict["image_h_w"] = (height, width)
            return edict(input_dict), edict(target_dict)

        num_target_views, num_views, bs = self.config.training.num_target_views, data_batch["c2w"].size(1), data_batch["image"].size(0)
        assert num_target_views < num_views, f"We have {num_views} views, but we want to select {num_target_views} target views. This is more than the total number of views we have."

        # Decide the target view indices
        is_inference = self.config.inference.get("if_inference", False)

        if target_has_input:
            # Randomly sample target views across all views
            index = torch.tensor([
                random.sample(range(num_views), num_target_views)
                for _ in range(bs)
            ], dtype=torch.long, device=data_batch["image"].device) # [b, num_target_views]

            # In inference mode, sort indices to preserve frame order (e.g., 40, 41, 42, ...)
            if is_inference:
                index = torch.sort(index, dim=1).values # [b, num_target_views]
        else:
            assert (
                self.config.training.num_input_views + num_target_views <= self.config.training.num_views
            ), f"We have {self.config.training.num_views} views in total, but we want to select {self.config.training.num_input_views} input views and {num_target_views} target views. This is more than the total number of views we have."

            index = torch.tensor([
                [self.config.training.num_views - 1 - j for j in range(num_target_views)]
                for _ in range(bs)
            ], dtype=torch.long, device=data_batch["image"].device)
            index = torch.sort(index, dim=1).values # [b, num_target_views]


        for key, value in data_batch.items():
            if not isinstance(value, torch.Tensor):
                input_dict[key] = value
                target_dict[key] = value
                continue
            if key.startswith("chain_"):
                # Chain tensors use shape [b, steps, v, ...].
                # Keep steps dimension intact and apply view split/gather on dim=2.
                if value.dim() >= 3:
                    input_dict[key] = value[:, :, :encoder_num_input_views, ...]
                    to_expand_dim = value.shape[3:]
                    expanded_index = index.view(
                        index.shape[0], 1, index.shape[1], *(1,) * len(to_expand_dim)
                    ).expand(-1, value.shape[1], -1, *to_expand_dim)
                    target_dict[key] = torch.gather(value, dim=2, index=expanded_index)
                    # Keep an explicit alias for per-pass supervision logging/loss code paths.
                    if key == "chain_relit_images":
                        input_dict["pass_relit_images"] = input_dict[key]
                        target_dict["pass_relit_images"] = target_dict[key]
                else:
                    input_dict[key] = value
                    target_dict[key] = value
                continue
            input_dict[key] = value[:, :encoder_num_input_views, ...]

            to_expand_dim = value.shape[2:] # [b, v, (value dim)] -> [value dim], e.g. [c, h, w] or [4] or [4, 4]
            expanded_index = index.view(index.shape[0], index.shape[1], *(1,) * len(to_expand_dim)).expand(-1, -1, *to_expand_dim)

            # Don't have target image supervision
            if key == "image" and not has_target_image:
                continue
            else:
                target_dict[key] = torch.gather(value, dim=1, index=expanded_index)

        height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
        input_dict["image_h_w"] = (height, width)
        target_dict["image_h_w"] = (height, width)

        input_dict, target_dict = edict(input_dict), edict(target_dict)

        return input_dict, target_dict



    @torch.no_grad()
    def forward(self, data_batch, has_target_image=True, target_has_input=True, compute_rays=True, force_full_target=False):
        """
        Preprocesses the input data batch and (optionally) computes ray_o and ray_d.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            has_target_image (bool): If True, target views have image supervision.
            target_has_input (bool): If True, target views can be sampled from input views.
            compute_rays (bool): If True, compute ray_o and ray_d.

        Returns:
            Input and Target data_batch (dict): Contains processed tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
                - 'ray_o' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'ray_d' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'image_h_w' (tuple): (height, width)
        """
        input_dict, target_dict = self.fetch_views(data_batch, has_target_image=has_target_image, target_has_input=target_has_input, force_full_target=force_full_target)

        if compute_rays:
            for dict in [input_dict, target_dict]:
                c2w = dict["c2w"]
                fxfycxcy = dict["fxfycxcy"]
                image_height, image_width = dict["image_h_w"]

                ray_o, ray_d = self.compute_rays(c2w, fxfycxcy, image_height, image_width, device=data_batch["image"].device)
                dict["ray_o"], dict["ray_d"] = ray_o, ray_d

        # Compute environment map directions for input and target.
        # If envmap_c2w is provided (e.g. Stanford ORB), use it instead of the
        # camera c2w so that the env directions match the envmap's own viewpoint.
        for d in [input_dict, target_dict]:
            env_c2w = d.get("envmap_c2w", None)
            if env_c2w is None:
                env_c2w = d["c2w"]
            env_dir = self.compute_env_dir(env_c2w, envmap_h=256, envmap_w=512, device=data_batch["image"].device)
            d["env_dir"] = env_dir

        return input_dict, target_dict
