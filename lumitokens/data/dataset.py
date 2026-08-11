# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
# Modified for the LumiTokens public inference release, 2026.

import random
import re
import traceback
import os
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F



class Dataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.og_dataset_base = self._normalize_base_path(
            self.config.training.get("og_dataset_base", None)
        )
        self.local_dataset_base = self._normalize_base_path(
            self.config.training.get("local_dataset_base", None)
        )

        try:
            # Support joint train/test loading by allowing multiple full_list files:
            # - training.dataset_path: single path or comma-separated paths
            # - training.extra_dataset_paths: optional list / comma-separated string
            dataset_list_paths = []
            dataset_path_raw = self.config.training.dataset_path
            if isinstance(dataset_path_raw, str):
                dataset_list_paths.extend([p.strip() for p in dataset_path_raw.split(",") if p.strip()])
            else:
                dataset_list_paths.append(str(dataset_path_raw).strip())

            extra_dataset_paths = self.config.training.get("extra_dataset_paths", [])
            if isinstance(extra_dataset_paths, str):
                dataset_list_paths.extend([p.strip() for p in extra_dataset_paths.split(",") if p.strip()])
            elif isinstance(extra_dataset_paths, (list, tuple)):
                dataset_list_paths.extend([str(p).strip() for p in extra_dataset_paths if str(p).strip()])
            elif extra_dataset_paths:
                dataset_list_paths.append(str(extra_dataset_paths).strip())

            # Keep order while deduplicating list files.
            seen_list_files = set()
            dataset_list_paths = [
                self._remap_path(p) for p in dataset_list_paths
                if not (p in seen_list_files or seen_list_files.add(p))
            ]

            all_scene_paths = []
            seen_scene_paths = set()
            for list_path in dataset_list_paths:
                with open(list_path, 'r') as f:
                    scene_paths = [self._remap_path(path.strip()) for path in f.read().splitlines() if path.strip()]
                for scene_path in scene_paths:
                    if scene_path not in seen_scene_paths:
                        seen_scene_paths.add(scene_path)
                        all_scene_paths.append(scene_path)
            self.all_scene_paths = all_scene_paths

        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e

        # Condition reverse: swap input/relit sources so input comes from relit scene
        # and relit supervision + lighting comes from the current scene
        self.condition_reverse = self.config.training.get("condition_reverse", False)

        # Parse relight signal config early: scene filtering below depends on
        # whether point_light is an enabled lighting condition.
        self.use_relit_images = self.config.training.get("use_relit_images", True)
        self.relight_signals = self.config.training.get("relight_signals", ["envmap"])
        if isinstance(self.relight_signals, str):
            self.relight_signals = [self.relight_signals]
        self.relight_signals = list(self.relight_signals)
        self.use_relight_envmap = "envmap" in self.relight_signals
        self.use_relight_point_light = "point_light" in self.relight_signals

        # recon_only: encoder-decoder reconstruction only (no editor). In this mode we
        # never load relit/envmap/point-light fields and we do NOT filter the scene
        # list by lighting type (every scene in full_list is a valid input scene).
        self.recon_only = bool(self.config.training.get("recon_only", False))
        if self.recon_only:
            self.use_relit_images = False

        def _is_allowed_input_scene(scene_name: str) -> bool:
            # recon_only trains on all scenes regardless of lighting tag.
            if self.recon_only:
                return True
            scene_type = self._scene_lighting_type(scene_name)
            # envmap-only: strictly keep *_env_* and *_white_env_* scenes.
            if self.use_relight_envmap and (not self.use_relight_point_light):
                return scene_type == "envmap"
            # point-light-only: allow pure point-light and combined scenes.
            if self.use_relight_point_light and (not self.use_relight_envmap):
                return scene_type in ("point_light", "combined")
            # both signals enabled: allow all relightable lighting scene types.
            if self.use_relight_envmap and self.use_relight_point_light:
                return scene_type in ("envmap", "point_light", "combined")
            # Fallback default (no valid signal configured): behave like envmap-only.
            return scene_type == "envmap"

        # Filter scenes if whiteEnvInput is enabled
        # Only keep scenes ending with "white_env_0" as input images
        self.whiteEnvInput = self.config.training.get("whiteEnvInput", False)
        if self.whiteEnvInput:
            total_scenes_before = len(self.all_scene_paths)
            filtered_scene_paths = []
            for scene_path in self.all_scene_paths:
                # Extract scene name from path (e.g., "/path/to/metadata/scene_name.json" -> "scene_name")
                file_name = os.path.basename(scene_path)
                scene_name = file_name.replace('.json', '')
                # Only keep scenes ending with "white_env_0"
                if scene_name.endswith('_white_env_0'):
                    filtered_scene_paths.append(scene_path)
            self.all_scene_paths = filtered_scene_paths
            print(f"whiteEnvInput enabled: Filtered to {len(self.all_scene_paths)} scenes ending with 'white_env_0' (from {total_scenes_before} total)")
        elif self.condition_reverse:
            total_scenes_before = len(self.all_scene_paths)
            # Build per-metadata-dir index of object_id -> envmap scene names
            # so we can discard scenes that have zero envmap relit candidates.
            _envmap_by_dir = {}  # metadata_dir -> {object_id -> set(envmap scene names)}
            for scene_path in self.all_scene_paths:
                metadata_dir = os.path.dirname(scene_path)
                if metadata_dir not in _envmap_by_dir:
                    _envmap_by_dir[metadata_dir] = {}
                    for fn in os.listdir(metadata_dir):
                        if not fn.endswith('.json'):
                            continue
                        cand_name = fn[:-5]
                        if self._scene_lighting_type(cand_name) == "envmap":
                            cand_obj = self._extract_object_id(cand_name)
                            _envmap_by_dir[metadata_dir].setdefault(cand_obj, set()).add(cand_name)

            filtered_scene_paths = []
            for scene_path in self.all_scene_paths:
                metadata_dir = os.path.dirname(scene_path)
                scene_name = os.path.basename(scene_path).replace('.json', '')
                if not _is_allowed_input_scene(scene_name):
                    continue
                object_id = self._extract_object_id(scene_name)
                envmap_candidates = _envmap_by_dir[metadata_dir].get(object_id, set()) - {scene_name}
                # combined_candidates = _combined_by_dir[metadata_dir].get(object_id, set()) - {scene_name}
                # point_light_candidates = _point_light_by_dir[metadata_dir].get(object_id, set()) - {scene_name}
                if envmap_candidates:
                    filtered_scene_paths.append(scene_path)
            self.all_scene_paths = filtered_scene_paths
            print(f"condition_reverse enabled: Filtered to {len(self.all_scene_paths)} scenes with envmap relit candidates (from {total_scenes_before} total)")
        else:
            total_scenes_before = len(self.all_scene_paths)
            filtered_scene_paths = []
            for scene_path in self.all_scene_paths:
                file_name = os.path.basename(scene_path)
                scene_name = file_name.replace('.json', '')
                if _is_allowed_input_scene(scene_name):
                    filtered_scene_paths.append(scene_path)
            self.all_scene_paths = filtered_scene_paths
            print(f"whiteEnvInput disabled: Filtered to {len(self.all_scene_paths)} scenes with valid lighting tags (from {total_scenes_before} total)")

        self.inference = self.config.inference.get("if_inference", False)
        self.render_all_views = self.config.inference.get("render_all_views", False)
        self.load_all_frames = bool(self.config.training.get("load_all_frames", False))
        # Inference-time random contiguous-chunk sampling (bypasses view_idx json).
        self.random_chunk_sampling = bool(self.config.inference.get("random_chunk_sampling", False))
        random_chunk_seed = self.config.inference.get("random_chunk_seed", None)
        self._chunk_rng = random.Random(random_chunk_seed) if random_chunk_seed is not None else random
        # Load file that specifies the input and target view indices to use for inference
        if self.inference and (not self.load_all_frames) and (not self.random_chunk_sampling):
            self.view_idx_list = dict()
            if self.config.inference.get("view_idx_file_path", None) is not None:
                if os.path.exists(self.config.inference.view_idx_file_path):
                    with open(self.config.inference.view_idx_file_path, 'r') as f:
                        self.view_idx_list = json.load(f)
                        # filter out None values, i.e. scenes that don't have specified input and targetviews
                        self.view_idx_list_filtered = [k for k, v in self.view_idx_list.items() if v is not None]
                    filtered_scene_paths = []
                    for scene in self.all_scene_paths:
                        file_name = scene.split("/")[-1]
                        scene_name = file_name.split(".")[0]
                        if scene_name in self.view_idx_list_filtered:
                            filtered_scene_paths.append(scene)

                    self.all_scene_paths = filtered_scene_paths
        else:
            self.view_idx_list = dict()

        # relight signal flags are parsed earlier (used for scene filtering above).
        self.point_light_num_rays = int(self.config.training.get("point_light_num_rays", 1024))
        self.exclude_white_env0_from_relit_sampling = bool(
            self.config.training.get("exclude_white_env0_from_relit_sampling", False)
        )

        # Check if we should load albedo images
        self.use_albedos = self.config.training.get("use_albedos", False)

        # Check if we should use white_env_0 scene images as albedo instead of loading from albedos folder
        self.white_env_as_albedo = self.config.training.get("white_env_as_albedo", False)


    def _normalize_base_path(self, path):
        if path is None:
            return None
        path = str(path).strip()
        if len(path) == 0:
            return None
        return os.path.normpath(path)

    def _remap_path(self, path):
        if path is None:
            return None
        path = str(path)
        if len(path) == 0:
            return path
        if self.og_dataset_base is None or self.local_dataset_base is None:
            return path

        norm_path = os.path.normpath(path)
        og_base = self.og_dataset_base

        if norm_path == og_base or norm_path.startswith(og_base + os.sep):
            rel = os.path.relpath(norm_path, og_base)
            if rel == ".":
                return self.local_dataset_base
            return os.path.join(self.local_dataset_base, rel)
        return path

    def __len__(self):
        return len(self.all_scene_paths)

    def _scene_lighting_type(self, scene_name: str) -> str:
        if "_white_env_" in scene_name or "_env_" in scene_name:
            return "envmap"
        if "_white_pl_" in scene_name or "_rgb_pl_" in scene_name:
            return "point_light"
        if "_multi_pl_" in scene_name:
            return "point_light"
        if "_area_" in scene_name:
            return "point_light"
        if "_combined_" in scene_name:
            return "combined"
        return "other"

    def _extract_object_id(self, scene_name: str) -> str:
        split_tags = ["_white_env_", "_env_", "_white_pl_", "_rgb_pl_",
                       "_multi_pl_", "_area_", "_combined_"]
        for tag in split_tags:
            idx = scene_name.rfind(tag)
            if idx != -1:
                return scene_name[:idx]
        # Fallback for unknown naming
        return scene_name.rsplit("_", 1)[0]

    def _is_white_env0_scene(self, scene_name: str) -> bool:
        return scene_name.endswith("_white_env_0")

    def _extract_env_base_id(self, scene_name: str):
        """
        Return a signature for the base lighting environment, ignoring any
        rotation/variation suffix. Used to ensure the source scene and the
        sampled relit scene do NOT come from the same ``*_env_{k}_*`` base env.

        Examples:
            obj_env_0_5      -> obj_env_0
            obj_env_2        -> obj_env_2
            obj_white_env_0  -> obj_white_env_0
        Returns None when the name does not match an env pattern.
        """
        m = re.match(r"^(.+)_white_env_(\d+)$", scene_name)
        if m:
            return f"{m.group(1)}_white_env_{m.group(2)}"
        m = re.match(r"^(.+)_env_(\d+)_(\d+)$", scene_name)
        if m:
            return f"{m.group(1)}_env_{m.group(2)}"
        m = re.match(r"^(.+)_env_(\d+)$", scene_name)
        if m:
            return f"{m.group(1)}_env_{m.group(2)}"
        return None

    def _load_point_light_rays(self, base_dir: str, scene_name: str, num_views: int) -> torch.Tensor:
        base_dir = self._remap_path(base_dir)
        rays_path = self._remap_path(os.path.join(base_dir, "point_light_rays", f"{scene_name}.npy"))
        if not os.path.exists(rays_path):
            raise FileNotFoundError(f"Point light rays file not found: {rays_path}")
        rays = np.load(rays_path)  # [N, 10]
        if rays.ndim != 2 or rays.shape[1] != 10:
            raise ValueError(f"Point light rays should have shape [N,10], got {rays.shape} at {rays_path}")

        sampled = []
        n_total = rays.shape[0]
        for _ in range(num_views):
            replace = n_total < self.point_light_num_rays
            sampled_idx = np.random.choice(n_total, size=self.point_light_num_rays, replace=replace)
            sampled.append(torch.from_numpy(rays[sampled_idx]).float())  # [num_rays, 10]
        return torch.stack(sampled, dim=0)  # [v, num_rays, 10]

    def _load_scene_images_by_indices(self, metadata_dir, scene_name, image_indices):
        metadata_dir = self._remap_path(metadata_dir)
        relit_scene_path = self._remap_path(os.path.join(metadata_dir, scene_name + ".json"))
        if not os.path.exists(relit_scene_path):
            raise FileNotFoundError(f"Relit scene JSON not found: {relit_scene_path}")

        with open(relit_scene_path, "r") as f:
            relit_data_json = json.load(f)
        relit_frames = relit_data_json.get("frames", [])
        if len(relit_frames) <= max(image_indices):
            raise IndexError(
                f"Relit scene '{scene_name}' has only {len(relit_frames)} frames, "
                f"but need frame index {max(image_indices)}"
            )

        relit_image_paths = [self._remap_path(relit_frames[ic]["image_path"]) for ic in image_indices]
        missing_images = [img_path for img_path in relit_image_paths if not os.path.exists(img_path)]
        if missing_images:
            if len(missing_images) > 3:
                raise FileNotFoundError(
                    f"Missing relit image files for scene '{scene_name}': {missing_images[:3]}..."
                )
            raise FileNotFoundError(f"Missing relit image files: {missing_images}")

        relit_frames_chosen = [relit_frames[ic] for ic in image_indices]
        relit_images, _, _ = self.preprocess_frames(relit_frames_chosen, relit_image_paths)
        return relit_images

    def _load_envmaps_or_zeros(self, base_dir, scene_name, image_indices):
        base_dir = self._remap_path(base_dir)
        envmaps_dir = self._remap_path(os.path.join(base_dir, "envmaps", scene_name))
        env_ldr_list = []
        env_hdr_list = []
        missing_slots = []
        env_h = None
        env_w = None
        default_h, default_w = 256, 512

        for ic in image_indices:
            env_ldr_path = os.path.join(envmaps_dir, f"{ic:05d}_ldr.png")
            env_hdr_path = os.path.join(envmaps_dir, f"{ic:05d}_hdr.png")
            if (not os.path.exists(env_ldr_path)) or (not os.path.exists(env_hdr_path)):
                if env_h is None or env_w is None:
                    env_ldr_list.append(None)
                    env_hdr_list.append(None)
                    missing_slots.append(len(env_ldr_list) - 1)
                else:
                    env_ldr_list.append(torch.zeros(3, env_h, env_w, dtype=torch.float32))
                    env_hdr_list.append(torch.zeros(3, env_h, env_w, dtype=torch.float32))
                continue

            try:
                env_ldr_img = PIL.Image.open(env_ldr_path)
                env_ldr_img.load()
                env_ldr_array = np.array(env_ldr_img) / 255.0
                if len(env_ldr_array.shape) == 2:
                    env_ldr_array = np.stack([env_ldr_array, env_ldr_array, env_ldr_array], axis=2)
                elif env_ldr_array.shape[2] == 4:
                    rgb = env_ldr_array[:, :, :3]
                    alpha = env_ldr_array[:, :, 3:4]
                    env_ldr_array = rgb * alpha + (1.0 - alpha) * 1.0
                elif env_ldr_array.shape[2] != 3:
                    env_ldr_array = env_ldr_array[:, :, :3]
                env_ldr_tensor = torch.from_numpy(env_ldr_array).permute(2, 0, 1).float()

                env_hdr_img = PIL.Image.open(env_hdr_path)
                env_hdr_img.load()
                env_hdr_array = np.array(env_hdr_img) / 255.0
                if len(env_hdr_array.shape) == 2:
                    env_hdr_array = np.stack([env_hdr_array, env_hdr_array, env_hdr_array], axis=2)
                elif env_hdr_array.shape[2] == 4:
                    rgb = env_hdr_array[:, :, :3]
                    alpha = env_hdr_array[:, :, 3:4]
                    env_hdr_array = rgb * alpha + (1.0 - alpha) * 1.0
                elif env_hdr_array.shape[2] != 3:
                    env_hdr_array = env_hdr_array[:, :, :3]
                env_hdr_tensor = torch.from_numpy(env_hdr_array).permute(2, 0, 1).float()

                if env_h is None or env_w is None:
                    env_h, env_w = env_ldr_tensor.shape[1], env_ldr_tensor.shape[2]
                    for slot_idx in missing_slots:
                        env_ldr_list[slot_idx] = torch.zeros(3, env_h, env_w, dtype=torch.float32)
                        env_hdr_list[slot_idx] = torch.zeros(3, env_h, env_w, dtype=torch.float32)
                    missing_slots = []

                if env_ldr_tensor.shape[1] != env_h or env_ldr_tensor.shape[2] != env_w:
                    env_ldr_tensor = torch.nn.functional.interpolate(
                        env_ldr_tensor.unsqueeze(0),
                        size=(env_h, env_w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)

                if env_hdr_tensor.shape[1] != env_h or env_hdr_tensor.shape[2] != env_w:
                    env_hdr_tensor = torch.nn.functional.interpolate(
                        env_hdr_tensor.unsqueeze(0),
                        size=(env_h, env_w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)

                env_ldr_list.append(env_ldr_tensor)
                env_hdr_list.append(env_hdr_tensor)
            except Exception:
                if env_h is None or env_w is None:
                    env_ldr_list.append(None)
                    env_hdr_list.append(None)
                    missing_slots.append(len(env_ldr_list) - 1)
                else:
                    env_ldr_list.append(torch.zeros(3, env_h, env_w, dtype=torch.float32))
                    env_hdr_list.append(torch.zeros(3, env_h, env_w, dtype=torch.float32))

        if env_h is None or env_w is None:
            env_h, env_w = default_h, default_w

        if missing_slots:
            for slot_idx in missing_slots:
                env_ldr_list[slot_idx] = torch.zeros(3, env_h, env_w, dtype=torch.float32)
                env_hdr_list[slot_idx] = torch.zeros(3, env_h, env_w, dtype=torch.float32)

        return torch.stack(env_ldr_list, dim=0), torch.stack(env_hdr_list, dim=0)


    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            cur_image_path = self._remap_path(cur_image_path)
            try:
                # Try to open the image file
                image = PIL.Image.open(cur_image_path)
                # Verify the image is valid by attempting to load it
                image.load()
            except (PIL.UnidentifiedImageError, OSError, IOError, Exception) as e:
                # If image file is corrupted or cannot be identified, raise an error
                error_msg = f"Error loading image file '{cur_image_path}': {type(e).__name__}: {str(e)}"
                print(error_msg)
                raise RuntimeError(error_msg) from e

            original_image_w, original_image_h = image.size

            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            # if torch.distributed.get_rank() == 0:
            #     import ipdb; ipdb.set_trace()

            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0

            # Handle RGBA images: alpha blend with white background
            if image.shape[2] == 4:  # RGBA image
                rgb = image[:, :, :3]  # Extract RGB channels
                alpha = image[:, :, 3:4]  # Extract alpha channel [h, w, 1]
                # Alpha blend with white background: RGB * alpha + (1 - alpha) * white
                # white = 1.0 (normalized)
                image = rgb * alpha + (1.0 - alpha) * 1.0
            elif image.shape[2] == 3:  # RGB image
                image = image
            else:
                # Convert grayscale or other formats to RGB
                if len(image.shape) == 2:  # Grayscale
                    image = np.stack([image, image, image], axis=2)
                else:
                    # Take first 3 channels if more than 3
                    image = image[:, :, :3]

            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws


        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def view_selector(self, frames):
        # print(f"frames: {len(frames)}")
        # print(f"num_views: {self.config.training.num_views}")
        if len(frames) < self.config.training.num_views:
            print(f"Not enough frames to sample")
            return None
        # sample view candidates
        view_selector_config = self.config.training.view_selector
        min_frame_dist = view_selector_config.get("min_frame_dist", 25)
        max_frame_dist = min(len(frames) - 1, view_selector_config.get("max_frame_dist", 100))
        if max_frame_dist <= min_frame_dist:
            print(f"max_frame_dist: {max_frame_dist}")
            print(f"min_frame_dist: {min_frame_dist}")
            print(f"max_frame_dist <= min_frame_dist")
            return None
        frame_dist = random.randint(min_frame_dist, max_frame_dist)
        if len(frames) <= frame_dist:
            print(f"len(frames): {len(frames)}")
            print(f"frame_dist: {frame_dist}")
            print(f"len(frames) <= frame_dist")
            return None
        start_frame = random.randint(0, len(frames) - frame_dist - 1)
        end_frame = start_frame + frame_dist
        # Check if we have enough frames in the range to sample
        # We need num_views-2 samples from range(start_frame+1, end_frame)
        # which has size: end_frame - start_frame - 1 = frame_dist - 1
        num_samples_needed = self.config.training.num_views - 2
        available_range_size = end_frame - start_frame - 1
        if available_range_size < num_samples_needed:
            print(f"available_range_size: {available_range_size}")
            print(f"num_samples_needed: {num_samples_needed}")
            print(f"available_range_size < num_samples_needed")
            return None
        sampled_frames = random.sample(range(start_frame + 1, end_frame), num_samples_needed)
        image_indices = [start_frame, end_frame] + sampled_frames
        return image_indices

    def load_all_view_selector(self, num_frames):
        """Load all frames and select evenly spaced condition frames first."""
        num_input_views = int(self.config.training.num_input_views)
        if num_input_views <= 0:
            return None
        if num_frames < num_input_views:
            return None

        # Evenly sample condition frames across the whole scene.
        raw_positions = np.linspace(0, num_frames - 1, num=num_input_views)
        context_indices = []
        seen = set()
        for pos in raw_positions:
            idx = int(round(float(pos)))
            idx = max(0, min(num_frames - 1, idx))
            if idx not in seen:
                context_indices.append(idx)
                seen.add(idx)

        # Fill dedup gaps while preserving original frame order.
        if len(context_indices) < num_input_views:
            for idx in range(num_frames):
                if idx not in seen:
                    context_indices.append(idx)
                    seen.add(idx)
                if len(context_indices) >= num_input_views:
                    break

        if len(context_indices) < num_input_views:
            return None

        all_indices = list(range(num_frames))
        remaining_indices = [i for i in all_indices if i not in seen]
        return context_indices + remaining_indices

    def random_chunk_view_selector(self, num_frames):
        """Pick a random contiguous chunk of (n_context + n_target) frames, then place
        n_context context frames evenly within the chunk. Returns indices ordered as
        [context_frames..., target_frames...] (both ascending by frame id) so the
        downstream input/target split stays stable.
        """
        n_context = int(self.config.training.num_input_views)
        n_target = int(self.config.training.num_target_views)
        chunk_len = n_context + n_target
        if n_context <= 0 or chunk_len <= 0:
            return None
        if num_frames < chunk_len:
            return None

        start = self._chunk_rng.randint(0, num_frames - chunk_len)
        window = list(range(start, start + chunk_len))  # contiguous, json order

        # Evenly place context positions within the chunk.
        raw_positions = np.linspace(0, chunk_len - 1, num=n_context)
        ctx_pos = []
        seen_pos = set()
        for p in raw_positions:
            pi = int(round(float(p)))
            pi = max(0, min(chunk_len - 1, pi))
            if pi not in seen_pos:
                seen_pos.add(pi)
                ctx_pos.append(pi)

        # Fill any dedup gaps so we always get exactly n_context context frames.
        if len(ctx_pos) < n_context:
            for pi in range(chunk_len):
                if pi not in seen_pos:
                    seen_pos.add(pi)
                    ctx_pos.append(pi)
                if len(ctx_pos) >= n_context:
                    break

        ctx_pos = sorted(ctx_pos[:n_context])
        ctx_pos_set = set(ctx_pos)
        context_ids = [window[p] for p in ctx_pos]
        target_ids = [window[p] for p in range(chunk_len) if p not in ctx_pos_set]
        return context_ids + target_ids

    def __getitem__(self, idx, max_retries=10, _retry_count=0):
        """
        Get item from dataset with error handling and retry logic.

        Args:
            idx: Index of the item to retrieve
            max_retries: Maximum number of retries before giving up
            _retry_count: Internal counter for retry attempts
        """
        if _retry_count >= max_retries:
            # If we've exhausted retries, raise an error with helpful message
            error_msg = f"Failed to load data after {max_retries} retries. Last attempted index: {idx}"
            print(error_msg)
            raise RuntimeError(error_msg)

        try:
            scene_path = self._remap_path(self.all_scene_paths[idx].strip())

            # Check if file exists
            if not os.path.exists(scene_path):
                raise FileNotFoundError(f"Scene JSON file not found: {scene_path}")

            # Load scene JSON
            try:
                with open(scene_path, 'r') as f:
                    data_json = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                raise RuntimeError(f"Error reading/parsing scene JSON '{scene_path}': {type(e).__name__}: {str(e)}") from e

            frames = data_json.get("frames", [])
            if not frames:
                raise ValueError(f"No frames found in scene JSON: {scene_path}")

            scene_name = data_json.get("scene_name", f"scene_{idx}")
            # print(f"scene_name: {scene_name}")
            # print(f"frames: {len(frames)}")

            if self.load_all_frames:
                image_indices = self.load_all_view_selector(len(frames))
                if image_indices is None:
                    print(f"load_all_view_selector returned None for scene {scene_name} at index {idx}, trying another random index")
                    return self.__getitem__(random.randint(0, len(self) - 1), max_retries=max_retries, _retry_count=_retry_count + 1)
            elif self.inference and self.random_chunk_sampling:
                # Random contiguous chunk with evenly-placed context frames (bypass view_idx json).
                image_indices = self.random_chunk_view_selector(len(frames))
                if image_indices is None:
                    print(f"random_chunk_view_selector returned None for scene {scene_name} at index {idx}, trying another random index")
                    return self.__getitem__(random.randint(0, len(self) - 1), max_retries=max_retries, _retry_count=_retry_count + 1)
            elif self.inference and self.render_all_views:
                # Load ALL frames: context from view_idx (or first N), rest as targets
                if scene_name in self.view_idx_list:
                    context_indices = self.view_idx_list[scene_name]["context"]
                else:
                    context_indices = list(range(self.config.training.num_input_views))
                all_indices = list(range(len(frames)))
                remaining_indices = sorted([i for i in all_indices if i not in context_indices])
                image_indices = context_indices + remaining_indices
            elif self.inference and scene_name in self.view_idx_list:
                current_view_idx = self.view_idx_list[scene_name]
                image_indices = current_view_idx["context"] + current_view_idx["target"]
            else:
                # sample input and target views
                image_indices = self.view_selector(frames)
                if image_indices is None:
                    # Fallback: try another random index
                    print(f"view_selector returned None for scene {scene_name} at index {idx}, trying another random index")
                    return self.__getitem__(random.randint(0, len(self) - 1), max_retries=max_retries, _retry_count=_retry_count + 1)

            # Validate image indices
            if not image_indices or len(image_indices) == 0:
                raise ValueError(f"No valid image indices selected for scene {scene_name}")

            # Check if all indices are valid
            for ic in image_indices:
                if ic < 0 or ic >= len(frames):
                    raise IndexError(f"Image index {ic} out of range for scene {scene_name} (has {len(frames)} frames)")

            image_paths_chosen = [self._remap_path(frames[ic]["image_path"]) for ic in image_indices]
            frames_chosen = [frames[ic] for ic in image_indices]

            # Check if all image paths exist
            for img_path in image_paths_chosen:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file not found: {img_path}")

            # Preprocess frames (this may raise PIL.UnidentifiedImageError or other errors)
            input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)

            # centerize and scale the poses (for unbounded scenes)
            scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
            input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)


            # Load relit images and environment maps from a different env scene with the same object_id
            relit_images = None
            env_ldr = None
            env_hdr = None
            albedos = None
            chain_relit_images = None
            chain_env_ldr = None
            chain_env_hdr = None
            chain_num_steps = None
            chain_scene_names = None

            object_id = self._extract_object_id(scene_name)

            # Extract base directory from scene_path (e.g., ".../train/metadata/...")
            # This is needed for both relit images and albedo loading
            scene_path_dir = os.path.dirname(scene_path)
            base_dir = self._remap_path(os.path.dirname(scene_path_dir))  # Go up from metadata to train/test

            point_light_rays = None

            # Load relit images and lighting conditions only if configured
            if self.use_relit_images:

                # Find all scenes with the same object_id but different env_name
                metadata_dir = self._remap_path(os.path.join(base_dir, 'metadata'))
                if not os.path.exists(metadata_dir):
                    error_msg = f"Metadata directory not found: {metadata_dir}"
                    print(f"Error: {error_msg}")
                    raise FileNotFoundError(error_msg)

                all_scene_json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
                candidate_scenes_by_type = {"envmap": [], "point_light": [], "combined": []}
                for json_file in all_scene_json_files:
                    candidate_scene_name = json_file[:-5]  # Remove .json extension
                    if not (candidate_scene_name.startswith(object_id + '_') and candidate_scene_name != scene_name):
                        continue
                    if (
                        self.exclude_white_env0_from_relit_sampling
                        and self._is_white_env0_scene(candidate_scene_name)
                    ):
                        continue
                    scene_type = self._scene_lighting_type(candidate_scene_name)
                    # Skip "combined_*" scenes entirely when point_light is not an enabled
                    # signal (they carry a point-light component). Covers all candidate
                    # branches below, including condition_reverse fallback.
                    if scene_type == "combined" and not self.use_relight_point_light:
                        continue
                    if scene_type in candidate_scenes_by_type:
                        candidate_scenes_by_type[scene_type].append(candidate_scene_name)

                enabled_signal_types = []
                if self.use_relight_envmap:
                    enabled_signal_types.append("envmap")
                if self.use_relight_point_light:
                    enabled_signal_types.append("point_light")
                if len(enabled_signal_types) == 0:
                    enabled_signal_types = ["envmap"]

                candidate_scenes = []
                if self.condition_reverse:
                    # Respect enabled signal types even in reverse-conditioning mode.
                    if self.use_relight_envmap:
                        candidate_scenes = list(candidate_scenes_by_type["envmap"])
                    if len(candidate_scenes) == 0 and self.use_relight_point_light:
                        candidate_scenes = list(candidate_scenes_by_type["combined"])
                    if len(candidate_scenes) == 0 and self.use_relight_point_light:
                        candidate_scenes = list(candidate_scenes_by_type["point_light"])
                else:
                    for sig in enabled_signal_types:
                        candidate_scenes.extend(candidate_scenes_by_type[sig])
                    # Combined scenes mix a point-light component with envmap, so only
                    # use them as relit targets when point_light is an enabled signal.
                    if self.use_relight_point_light:
                        candidate_scenes.extend(candidate_scenes_by_type["combined"])

                # Ensure the relit scene comes from a DIFFERENT base env than the
                # source scene: exclude candidates that share the same *_env_{k}_*
                # base env (rotations of the same env should not be a relit target).
                source_env_base = self._extract_env_base_id(scene_name)
                if source_env_base is not None:
                    different_env_candidates = [
                        c for c in candidate_scenes
                        if self._extract_env_base_id(c) != source_env_base
                    ]
                    # Fall back to the unfiltered list only if filtering removed everything.
                    if different_env_candidates:
                        candidate_scenes = different_env_candidates

                if not candidate_scenes:
                    error_msg = (
                        f"No candidate relit scenes found for object_id '{object_id}' "
                        f"with relight_signals={enabled_signal_types} (current scene: {scene_name})"
                    )
                    print(f"Error: {error_msg}")
                    raise ValueError(error_msg)

                # A release example may pin its lighting scene for deterministic
                # regression testing. Normal dataset usage keeps random sampling.
                fixed_relit_scene = self.config.inference.get("relight_scene_name", None)
                if fixed_relit_scene is not None:
                    if fixed_relit_scene not in candidate_scenes:
                        raise ValueError(
                            f"Configured relight scene '{fixed_relit_scene}' is not a valid "
                            f"candidate for source scene '{scene_name}'."
                        )
                    relit_scene_name = fixed_relit_scene
                else:
                    relit_scene_name = random.choice(candidate_scenes)
                # print(f"chose relit scene: {relit_scene_name}")
                relit_scene_path = self._remap_path(os.path.join(metadata_dir, relit_scene_name + '.json'))

                # Load relit scene JSON
                if not os.path.exists(relit_scene_path):
                    error_msg = f"Relit scene JSON not found: {relit_scene_path}"
                    print(f"Error: {error_msg}")
                    raise FileNotFoundError(error_msg)

                with open(relit_scene_path, 'r') as f:
                    relit_data_json = json.load(f)
                relit_frames = relit_data_json.get("frames", [])

                # Check if relit scene has enough frames
                if len(relit_frames) <= max(image_indices):
                    error_msg = f"Relit scene '{relit_scene_name}' has only {len(relit_frames)} frames, but need frame index {max(image_indices)}"
                    print(f"Error: {error_msg}")
                    raise IndexError(error_msg)

                # Load relit images with same indices
                relit_image_paths = [self._remap_path(relit_frames[ic]["image_path"]) for ic in image_indices]
                relit_frames_chosen = [relit_frames[ic] for ic in image_indices]

                # Check if all relit image paths exist
                missing_images = [img_path for img_path in relit_image_paths if not os.path.exists(img_path)]
                if missing_images:
                    error_msg = f"Missing relit image files for scene '{relit_scene_name}': {missing_images[:3]}..." if len(missing_images) > 3 else f"Missing relit image files: {missing_images}"
                    print(f"Error: {error_msg}")
                    raise FileNotFoundError(error_msg)

                # Load relit images
                relit_images, _, _ = self.preprocess_frames(relit_frames_chosen, relit_image_paths)

                # Condition reverse: input images come from relit scene,
                # relit supervision + lighting come from the current scene
                if self.condition_reverse:
                    input_images, relit_images = relit_images, input_images
                    lighting_source_scene_name = scene_name
                else:
                    lighting_source_scene_name = relit_scene_name

                # Determine actual lighting type of the lighting source scene
                actual_lighting_type = self._scene_lighting_type(lighting_source_scene_name)

                # Determine whether this scene should provide envmaps / point-light rays.
                # For "combined" scenes we attempt both and gracefully fall back to zeros.
                should_load_envmap = actual_lighting_type == "envmap" or actual_lighting_type == "combined"
                should_load_point_light = actual_lighting_type == "point_light" or actual_lighting_type == "combined"

                # Load envmaps if enabled by relight_signals
                if self.use_relight_envmap:
                    envmaps_loaded = False
                    if should_load_envmap:
                        envmaps_dir = self._remap_path(os.path.join(base_dir, 'envmaps', lighting_source_scene_name))
                        # For pure envmap scenes the directory must exist; for combined it's optional
                        has_envmaps_dir = os.path.exists(envmaps_dir)
                        if has_envmaps_dir:
                            env_ldr_list = []
                            env_hdr_list = []
                            all_found = True
                            for ic in image_indices:
                                env_ldr_path = self._remap_path(os.path.join(envmaps_dir, f"{ic:05d}_ldr.png"))
                                env_hdr_path = self._remap_path(os.path.join(envmaps_dir, f"{ic:05d}_hdr.png"))
                                if not os.path.exists(env_ldr_path) or not os.path.exists(env_hdr_path):
                                    if actual_lighting_type == "envmap":
                                        raise FileNotFoundError(f"Environment map file not found: {env_ldr_path} or {env_hdr_path}")
                                    all_found = False
                                    break
                                try:
                                    env_ldr_img = PIL.Image.open(env_ldr_path)
                                    env_ldr_img.load()
                                    env_ldr_array = np.array(env_ldr_img) / 255.0
                                    if len(env_ldr_array.shape) == 2:
                                        env_ldr_array = np.stack([env_ldr_array, env_ldr_array, env_ldr_array], axis=2)
                                    elif env_ldr_array.shape[2] == 4:
                                        rgb = env_ldr_array[:, :, :3]
                                        alpha = env_ldr_array[:, :, 3:4]
                                        env_ldr_array = rgb * alpha + (1.0 - alpha) * 1.0
                                    elif env_ldr_array.shape[2] != 3:
                                        env_ldr_array = env_ldr_array[:, :, :3]
                                    env_ldr_tensor = torch.from_numpy(env_ldr_array).permute(2, 0, 1).float()

                                    env_hdr_img = PIL.Image.open(env_hdr_path)
                                    env_hdr_img.load()
                                    env_hdr_array = np.array(env_hdr_img) / 255.0
                                    if len(env_hdr_array.shape) == 2:
                                        env_hdr_array = np.stack([env_hdr_array, env_hdr_array, env_hdr_array], axis=2)
                                    elif env_hdr_array.shape[2] == 4:
                                        rgb = env_hdr_array[:, :, :3]
                                        alpha = env_hdr_array[:, :, 3:4]
                                        env_hdr_array = rgb * alpha + (1.0 - alpha) * 1.0
                                    elif env_hdr_array.shape[2] != 3:
                                        env_hdr_array = env_hdr_array[:, :, :3]
                                    env_hdr_tensor = torch.from_numpy(env_hdr_array).permute(2, 0, 1).float()

                                    env_ldr_list.append(env_ldr_tensor)
                                    env_hdr_list.append(env_hdr_tensor)
                                except Exception as e:
                                    if actual_lighting_type == "envmap":
                                        raise RuntimeError(
                                            f"Failed to load environment map for frame {ic} in scene '{lighting_source_scene_name}': {e}"
                                        ) from e
                                    all_found = False
                                    break

                            if all_found and len(env_ldr_list) == len(image_indices):
                                env_ldr = torch.stack(env_ldr_list, dim=0)
                                env_hdr = torch.stack(env_hdr_list, dim=0)
                                envmaps_loaded = True
                            elif actual_lighting_type == "envmap":
                                raise ValueError(
                                    f"Mismatch in environment map count for scene '{lighting_source_scene_name}'"
                                )

                    if not envmaps_loaded:
                        env_h, env_w = 256, 512
                        if len(candidate_scenes_by_type["envmap"]) > 0:
                            sample_env_scene = candidate_scenes_by_type["envmap"][0]
                            sample_envmaps_dir = self._remap_path(os.path.join(base_dir, 'envmaps', sample_env_scene))
                            if os.path.exists(sample_envmaps_dir):
                                sample_env_files = [f for f in os.listdir(sample_envmaps_dir) if f.endswith('_ldr.png')]
                                if sample_env_files:
                                    sample_env_path = self._remap_path(os.path.join(sample_envmaps_dir, sample_env_files[0]))
                                    sample_env_img = PIL.Image.open(sample_env_path)
                                    env_h, env_w = sample_env_img.size[1], sample_env_img.size[0]
                        env_ldr = torch.zeros(len(image_indices), 3, env_h, env_w, dtype=torch.float32).clone()
                        env_hdr = torch.zeros(len(image_indices), 3, env_h, env_w, dtype=torch.float32).clone()

                # Load point light rays if enabled by relight_signals
                if self.use_relight_point_light:
                    point_light_loaded = False
                    if should_load_point_light:
                        rays_path = self._remap_path(os.path.join(base_dir, "point_light_rays", f"{lighting_source_scene_name}.npy"))
                        if os.path.exists(rays_path):
                            point_light_rays = self._load_point_light_rays(
                                base_dir=base_dir,
                                scene_name=lighting_source_scene_name,
                                num_views=len(image_indices),
                            )
                            point_light_loaded = True
                        elif actual_lighting_type == "point_light":
                            raise FileNotFoundError(f"Point light rays file not found: {rays_path}")

                    if not point_light_loaded:
                        point_light_rays = torch.zeros(
                            len(image_indices),
                            self.point_light_num_rays,
                            10,
                            dtype=torch.float32
                        ).clone()

                multi_edit_cfg = self.config.training.get("multi_edit", {})
                multi_edit_enable = bool(multi_edit_cfg.get("enable", False))
                if (
                    multi_edit_enable
                    and self.use_relight_envmap
                    and (not self.use_relight_point_light)
                    and (not self.condition_reverse)
                ):
                    max_steps = int(multi_edit_cfg.get("max_steps", 3))
                    max_steps = max(1, max_steps)
                    insufficient_chain_policy = multi_edit_cfg.get("insufficient_chain_policy", "resample")
                    if len(candidate_scenes) < max_steps:
                        # If max_steps exceeds available lighting variations,
                        # sample with replacement so the same variation can be reused.
                        sampled_chain = [random.choice(candidate_scenes) for _ in range(max_steps)]
                    else:
                        remaining_scenes = [s for s in candidate_scenes if s != relit_scene_name]
                        if max_steps > 1 and len(remaining_scenes) < (max_steps - 1):
                            if insufficient_chain_policy == "resample":
                                return self.__getitem__(
                                    random.randint(0, len(self) - 1),
                                    max_retries=max_retries,
                                    _retry_count=_retry_count + 1,
                                )
                            if insufficient_chain_policy == "sample_with_replacement":
                                sampled_chain = [relit_scene_name]
                                sampled_chain.extend(
                                    random.choices(candidate_scenes, k=max_steps - 1)
                                )
                            else:
                                raise ValueError(
                                    f"Not enough unique relit scenes for multi_edit chain after first sample, "
                                    f"need {max_steps - 1}, got {len(remaining_scenes)} "
                                    f"(scene: {scene_name}, object_id: {object_id})"
                                )
                        else:
                            sampled_chain = [relit_scene_name]
                            if max_steps > 1:
                                sampled_chain.extend(random.sample(remaining_scenes, max_steps - 1))

                    chain_relit_images_list = []
                    chain_env_ldr_list = []
                    chain_env_hdr_list = []
                    for chain_scene_name in sampled_chain:
                        chain_relit_images_list.append(
                            self._load_scene_images_by_indices(
                                metadata_dir=metadata_dir,
                                scene_name=chain_scene_name,
                                image_indices=image_indices,
                            )
                        )
                        cur_env_ldr, cur_env_hdr = self._load_envmaps_or_zeros(
                            base_dir=base_dir,
                            scene_name=chain_scene_name,
                            image_indices=image_indices,
                        )
                        chain_env_ldr_list.append(cur_env_ldr)
                        chain_env_hdr_list.append(cur_env_hdr)

                    chain_relit_images = torch.stack(chain_relit_images_list, dim=0)  # [steps, v, 3, h, w]
                    chain_env_ldr = torch.stack(chain_env_ldr_list, dim=0)            # [steps, v, 3, eh, ew]
                    chain_env_hdr = torch.stack(chain_env_hdr_list, dim=0)            # [steps, v, 3, eh, ew]
                    chain_num_steps = torch.tensor(max_steps, dtype=torch.long)
                    chain_scene_names = sampled_chain
            else:
                # Skip loading relit signals if not configured
                relit_images = None
                env_ldr = None
                env_hdr = None
                point_light_rays = None

            # Load albedo images
            # Only load if configured to do so
            if self.use_albedos:
                if self.white_env_as_albedo:
                    # Load albedo from white_env_0 scene instead of albedos folder
                    white_env_scene_name = object_id + '_white_env_0'
                    metadata_dir = self._remap_path(os.path.join(base_dir, 'metadata'))
                    if not os.path.exists(metadata_dir):
                        error_msg = f"Metadata directory not found: {metadata_dir}"
                        print(f"Error: {error_msg}")
                        raise FileNotFoundError(error_msg)

                    white_env_scene_path = self._remap_path(os.path.join(metadata_dir, white_env_scene_name + '.json'))
                    if not os.path.exists(white_env_scene_path):
                        error_msg = f"White env scene JSON not found: {white_env_scene_path}"
                        print(f"Error: {error_msg}")
                        raise FileNotFoundError(error_msg)

                    # Load white env scene JSON
                    with open(white_env_scene_path, 'r') as f:
                        white_env_data_json = json.load(f)
                    white_env_frames = white_env_data_json.get("frames", [])

                    # Check if white env scene has enough frames
                    if len(white_env_frames) <= max(image_indices):
                        error_msg = f"White env scene '{white_env_scene_name}' has only {len(white_env_frames)} frames, but need frame index {max(image_indices)}"
                        print(f"Error: {error_msg}")
                        raise IndexError(error_msg)

                    # Load white env images with same indices
                    white_env_image_paths = [self._remap_path(white_env_frames[ic]["image_path"]) for ic in image_indices]
                    white_env_frames_chosen = [white_env_frames[ic] for ic in image_indices]

                    # Check if all white env image paths exist
                    missing_images = [img_path for img_path in white_env_image_paths if not os.path.exists(img_path)]
                    if missing_images:
                        error_msg = f"Missing white env image files for scene '{white_env_scene_name}': {missing_images[:3]}..." if len(missing_images) > 3 else f"Missing white env image files: {missing_images}"
                        print(f"Error: {error_msg}")
                        raise FileNotFoundError(error_msg)

                    # Load and preprocess white env images as albedo (using same preprocessing as input images)
                    albedos, _, _ = self.preprocess_frames(white_env_frames_chosen, white_env_image_paths)
                else:
                    # Load albedo images from albedos folder (shared across all scenes with same object_id)
                    albedos_dir = self._remap_path(os.path.join(base_dir, 'albedos', object_id))
                    if os.path.exists(albedos_dir):
                        albedo_list = []
                        for ic in image_indices:
                            albedo_path = self._remap_path(os.path.join(albedos_dir, f"{ic:05d}.png"))

                            if not os.path.exists(albedo_path):
                                error_msg = f"Albedo file not found: {albedo_path}"
                                print(f"Error: {error_msg}")
                                raise FileNotFoundError(error_msg)

                            try:
                                albedo_img = PIL.Image.open(albedo_path)
                                albedo_img.load()

                                # Apply same preprocessing as input images (resize, crop, etc.)
                                # Get original dimensions
                                original_albedo_w, original_albedo_h = albedo_img.size
                                resize_h = self.config.model.image_tokenizer.image_size
                                patch_size = self.config.model.image_tokenizer.patch_size
                                square_crop = self.config.training.get("square_crop", False)

                                resize_w = int(resize_h / original_albedo_h * original_albedo_w)
                                resize_w = int(round(resize_w / patch_size) * patch_size)

                                # Resize albedo to match input image size
                                albedo_img = albedo_img.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
                                if square_crop:
                                    min_size = min(resize_h, resize_w)
                                    start_h = (resize_h - min_size) // 2
                                    start_w = (resize_w - min_size) // 2
                                    albedo_img = albedo_img.crop((start_w, start_h, start_w + min_size, start_h + min_size))

                                # Convert to numpy array
                                albedo_array = np.array(albedo_img) / 255.0

                                # Handle different image formats
                                if len(albedo_array.shape) == 2:
                                    # Grayscale: convert to RGB
                                    albedo_array = np.stack([albedo_array, albedo_array, albedo_array], axis=2)
                                elif albedo_array.shape[2] == 4:
                                    # RGBA: alpha blend with white background
                                    rgb = albedo_array[:, :, :3]
                                    alpha = albedo_array[:, :, 3:4]
                                    albedo_array = rgb * alpha + (1.0 - alpha) * 1.0
                                elif albedo_array.shape[2] == 3:
                                    # RGB: use as is
                                    pass
                                else:
                                    # More than 4 channels: take first 3
                                    albedo_array = albedo_array[:, :, :3]

                                albedo_tensor = torch.from_numpy(albedo_array).permute(2, 0, 1).float()
                                albedo_list.append(albedo_tensor)
                            except Exception as e:
                                error_msg = f"Failed to load albedo for frame {ic} in object_id '{object_id}': {type(e).__name__}: {str(e)}"
                                print(f"Error: {error_msg}")
                                traceback.print_exc()
                                raise RuntimeError(error_msg) from e

                        if len(albedo_list) != len(image_indices):
                            error_msg = f"Mismatch in albedo count: expected {len(image_indices)}, got {len(albedo_list)}"
                            print(f"Error: {error_msg}")
                            raise ValueError(error_msg)

                        albedos = torch.stack(albedo_list, dim=0)  # [v, 3, h, w]
                    else:
                        # Albedo directory doesn't exist, set to None
                        error_msg = f"Albedo directory not found: {albedos_dir}"
                        print(f"Error: {error_msg}")
                        raise FileNotFoundError(error_msg)
            else:
                # Skip loading albedo if not configured
                albedos = None

            image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
            scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
            indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

            result_dict = {
                "image": input_images,
                "c2w": input_c2ws,
                "fxfycxcy": input_intrinsics,
                "index": indices,
                "scene_name": scene_name
            }

            # Add optional relit images and environment maps
            # Always include these keys (even if None) to ensure consistent batch structure
            # This prevents KeyError during DataLoader collation when some samples have these fields and others don't
            if self.use_relit_images:
                result_dict["relit_images"] = relit_images
                result_dict["relit_scene_name"] = relit_scene_name  # Add relit scene name for tracking
                if self.use_relight_envmap:
                    result_dict["env_ldr"] = env_ldr
                    result_dict["env_hdr"] = env_hdr
                    if chain_relit_images is not None:
                        result_dict["chain_relit_images"] = chain_relit_images
                    if chain_env_ldr is not None:
                        result_dict["chain_env_ldr"] = chain_env_ldr
                    if chain_env_hdr is not None:
                        result_dict["chain_env_hdr"] = chain_env_hdr
                    if chain_num_steps is not None:
                        result_dict["chain_num_steps"] = chain_num_steps
                    if chain_scene_names is not None:
                        result_dict["chain_scene_names"] = chain_scene_names
                if self.use_relight_point_light:
                    result_dict["point_light_rays"] = point_light_rays
            if self.use_albedos:
                result_dict["albedos"] = albedos

            return result_dict
        except (ValueError, IndexError, KeyError, FileNotFoundError, RuntimeError,
                PIL.UnidentifiedImageError, OSError, IOError, json.JSONDecodeError,
                torch.cuda.OutOfMemoryError) as e:
            # Fallback for any data loading errors
            # Try another random index instead
            error_type = type(e).__name__
            error_msg = f"Error loading scene at index {idx} (attempt {_retry_count + 1}/{max_retries}): {error_type}: {str(e)}"
            print(error_msg)
            if _retry_count < 2:  # Only print traceback for first few attempts
                traceback.print_exc()

            # Try another random index
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx, max_retries=max_retries, _retry_count=_retry_count + 1)
