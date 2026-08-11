"""Render object-centric LumiTokens training data.

Migrated from the LumiTokens data-generation repository and adapted to use
portable, command-line supplied dataset paths for the public release.
"""

import json
import hashlib
import math
import os
from dataclasses import dataclass
import random
import sys
from typing import Optional

import imageio
import numpy as np
import simple_parsing
import shutil
error_list = []

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


GENERATOR_VERSION = "enhanced_v2_3_2026-07-22"


PROFILE_CONFIGS = {
    "stanford": {
        "weight": 0.4,
        "fov_range": (19.5, 21.5),
        "rho_train": (0.34, 0.88),
        "rho_test": (0.34, 0.88),
        "rho_variation_fraction": 0.08,
        "target_extent_train": (0.32, 0.52),
        "target_extent_test": (0.32, 0.52),
        "theta_center_range": (62.0, 78.0),
        "theta_amplitude_range": (1.0, 6.0),
        "azimuth_span_range": (300.0, 355.0),
        "aim_sigma_deg": 0.8,
        "aim_clip_deg": 3.0,
        "roll_sigma_deg": 0.7,
        "roll_clip_deg": 2.0,
        "exposure_range": (-0.25, 0.65),
        "gamma_range": (0.96, 1.04),
        "white_balance_range": (0.96, 1.04),
        "noise_std_range": (0.0, 0.004),
    },
    "owl": {
        "weight": 0.4,
        "fov_range": (39.0, 43.0),
        "rho_train": (0.42, 0.92),
        "rho_test": (0.34, 0.78),
        "rho_variation_fraction": 0.15,
        "target_extent_train": (0.50, 0.80),
        "target_extent_test": (0.38, 0.62),
        "theta_center_range": (38.0, 78.0),
        "theta_amplitude_range": (8.0, 24.0),
        "azimuth_span_range": (220.0, 350.0),
        "aim_sigma_deg": 5.0,
        "aim_sigma_train_deg": 5.5,
        "aim_sigma_test_deg": 3.5,
        "aim_clip_deg": 14.0,
        "aim_clip_test_deg": 11.0,
        "roll_sigma_deg": 2.0,
        "roll_clip_deg": 6.0,
        "exposure_range": (-0.35, 0.85),
        "gamma_range": (0.92, 1.08),
        "white_balance_range": (0.92, 1.08),
        "noise_std_range": (0.001, 0.010),
    },
    "broad": {
        "weight": 0.2,
        "fov_range": (18.0, 50.0),
        "rho_train": (0.28, 0.88),
        "rho_test": (0.28, 0.82),
        "rho_variation_fraction": 0.20,
        "target_extent_train": (0.28, 0.75),
        "target_extent_test": (0.28, 0.70),
        "theta_center_range": (28.0, 82.0),
        "theta_amplitude_range": (4.0, 28.0),
        "azimuth_span_range": (140.0, 355.0),
        "aim_sigma_deg": 3.5,
        "aim_clip_deg": 15.0,
        "roll_sigma_deg": 3.5,
        "roll_clip_deg": 10.0,
        "exposure_range": (-0.75, 0.90),
        "gamma_range": (0.90, 1.10),
        "white_balance_range": (0.90, 1.10),
        "noise_std_range": (0.0, 0.012),
    },
}


def stable_int_seed(*parts) -> int:
    """Return a deterministic 31-bit seed that is stable across processes."""
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 31 - 1)


def choose_camera_profile(requested_profile: str, rng: random.Random) -> str:
    if requested_profile != "mixture":
        if requested_profile not in PROFILE_CONFIGS:
            raise ValueError(
                f"Unknown camera_profile={requested_profile!r}; expected mixture or "
                f"one of {sorted(PROFILE_CONFIGS)}"
            )
        return requested_profile

    names = list(PROFILE_CONFIGS)
    weights = [PROFILE_CONFIGS[name]["weight"] for name in names]
    return rng.choices(names, weights=weights, k=1)[0]


def atomic_write_json(path: str, value) -> None:
    """Write JSON through a temporary sibling so retries never see a partial file."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as file:
        json.dump(value, file, indent=4)
    os.replace(tmp_path, path)

@dataclass
class Options:
    """ 3D dataset rendering script """
    three_d_model_path: str = ''  # Set from models_root + CSV entry
    models_root: str = './data/objaverse/glbs'
    env_map_list_json: str = os.path.join(SCRIPT_DIR, 'assets/hdri/polyhaven_hdris.json')
    env_map_dir_path: str = './data/hdri'
    white_env_map_dir_path: str = './data/hdri'
    output_dir: str = './outputs/rendered_objects'
    num_views: int = 200  # Number of views
    num_test_views: int = 100  # Number of test views (trajectory views)
    num_white_pls: int = 0  # Number of white point lighting
    num_rgb_pls: int = 0  # Number of RGB point lighting
    num_multi_pls: int = 0  # Number of multi point lighting
    max_pl_num: int = 3  # Maximum number of point lights
    num_white_envs: int = 1  # Number of white env lighting
    num_env_lights: int = 5  # Number of env lighting
    num_area_lights: int = 0  # Number of area lights
    num_combined_lights: int = 0  # Number of combined lights (env + white point light)
    seed: Optional[int] = None  # Random seed
    num_view_groups: int = 1  # Number of view groups
    group_start: int = 0
    group_end: int = 10  # Group of models to render
    save_intrinsics: bool = True  # Whether to save intrinsics for each view
    csv_path: str = "object_ids.csv"  # CSV rows: shard, object UID
    rendered_dir_name: str = "rendered_dense"  # Name of the rendered output directory (replaces 'glbs' in dataset path)
    output_dir_override: Optional[str] = None  # Optional explicit base output directory
    camera_profile: str = "mixture"  # mixture, stanford, owl, or broad
    global_seed: int = 20260722
    render_resolution: int = 512
    cycles_samples: int = 64
    use_denoising: bool = True
    apply_photometric_augmentation: bool = True
    min_camera_clearance: float = 1.20  # Minimum distance in scene-radius units
    frame_margin: float = 0.92  # Conservative normalized half-frame extent
    max_boundary_touch_fraction: float = 0.05
    max_photometric_invalid_fraction: float = 0.10
    min_mask_area: float = 0.003
    max_mask_area: float = 0.85
    min_foreground_luma: float = 0.01
    max_foreground_luma: float = 0.98
    rho_min: float = 0.8  # Deprecated compatibility option; profiles define occupancy
    rho_max: float = 1.0  # Deprecated compatibility option; profiles define occupancy


def render_core(args: Options, groups_id = 0):
    import bpy
    from mathutils import Matrix, Vector

    from bpy_helper.camera import create_camera
    from bpy_helper.io import render_depth_map, mat2list, array2list, render_normal_map, render_albedo_map, transform_normals_to_camera_space
    from bpy_helper.light import create_point_light, set_env_light, create_area_light
    from bpy_helper.random import gen_random_pts_around_origin
    from bpy_helper.scene import import_3d_model, normalize_scene, reset_scene
    from bpy_helper.utils import stdout_redirected

    file_path = args.three_d_model_path
    object_uid = os.path.splitext(os.path.basename(file_path))[0]
    base_seed = args.global_seed if args.seed is None else args.seed
    scene_seed = stable_int_seed(base_seed, object_uid, groups_id, GENERATOR_VERSION)
    scene_rng = random.Random(scene_seed)
    random.seed(scene_seed)
    np.random.seed(scene_seed)

    selected_profile = choose_camera_profile(args.camera_profile, scene_rng)
    profile_config = PROFILE_CONFIGS[selected_profile]
    wb_min, wb_max = profile_config["white_balance_range"]
    photo_config = {
        "exposure_stops": scene_rng.uniform(*profile_config["exposure_range"]),
        "gamma": scene_rng.uniform(*profile_config["gamma_range"]),
        "white_balance": [scene_rng.uniform(wb_min, wb_max) for _ in range(3)],
        "noise_std": scene_rng.uniform(*profile_config["noise_std_range"]),
    }

    def render_rgb_and_hint(output_path, idx=0):
        bpy.context.scene.view_layers["ViewLayer"].material_override = None
        bpy.context.scene.render.image_settings.file_format = "PNG"
        frame_path = os.path.join(output_path, f"gt_{idx}.png")
        bpy.context.scene.render.filepath = frame_path
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        if not args.apply_photometric_augmentation:
            return

        # Apply a deterministic, sequence-level camera response while preserving
        # straight alpha. Per-frame sensor noise is deterministic but not shared.
        image = imageio.v3.imread(frame_path)
        if not np.issubdtype(image.dtype, np.integer):
            max_value = 1.0
        else:
            max_value = float(np.iinfo(image.dtype).max)
        rgb = image[..., :3].astype(np.float32) / max_value
        alpha = image[..., 3:4] if image.shape[-1] == 4 else None

        rgb *= 2.0 ** float(photo_config["exposure_stops"])
        rgb *= np.asarray(photo_config["white_balance"], dtype=np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = np.power(rgb, 1.0 / float(photo_config["gamma"]))

        noise_seed = stable_int_seed(
            scene_seed,
            os.path.relpath(output_path, res_dir),
            idx,
            "sensor",
        )
        noise_rng = np.random.RandomState(noise_seed)
        noise_std = float(photo_config["noise_std"])
        if noise_std > 0:
            rgb += noise_rng.normal(0.0, noise_std, size=rgb.shape).astype(np.float32)
        rgb = (np.clip(rgb, 0.0, 1.0) * max_value).round().astype(image.dtype)
        output = np.concatenate([rgb, alpha], axis=-1) if alpha is not None else rgb
        imageio.v3.imwrite(frame_path, output)

    def configure_blender():
        scene = bpy.context.scene
        scene.render.resolution_x = args.render_resolution
        scene.render.resolution_y = args.render_resolution
        scene.render.resolution_percentage = 100
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.cycles_samples
        scene.cycles.use_denoising = args.use_denoising
        try:
            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            scene.cycles.device = "GPU"
        except Exception as error:
            print(f"Warning: CUDA setup failed; falling back to the available Cycles device: {error}")

        # Fix color management instead of inheriting the user's Blender homefile.
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "Medium High Contrast"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.sequencer_colorspace_settings.name = "sRGB"

        bpy.context.scene.render.film_transparent = True
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        scene.render.image_settings.compression = 15

    def set_neutral_world_ambient(level=0.02):
        world = bpy.context.scene.world
        world.use_nodes = True
        background = world.node_tree.nodes.get("Background")
        if background is not None and not background.inputs["Color"].is_linked:
            background.inputs["Color"].default_value = (level, level, level, 1.0)
            background.inputs["Strength"].default_value = 1.0

    reset_scene()
    configure_blender()

    #& 1.preparing the scene
    #* 1.1 prepare the 3d model
    with stdout_redirected():
        import_3d_model(file_path)
    scale, offset = normalize_scene(use_bounding_sphere=True)
    # Preserve imported alpha, emission, transmission, and cutout materials. The
    # previous global material rewrite damaged precisely the real-like assets we
    # want the model to learn from.
    sampled_object_radius = 0.5
    post_normalize_scale = 1.0
    recenter_offset = [0.0, 0.0, 0.0]

    # Load env map list
    with open(args.env_map_list_json, "r") as file:
        env_map_list = json.load(file)

    # Render GT images & hints
    seed_view = stable_int_seed(scene_seed, "view")
    seed_white_pl = stable_int_seed(scene_seed, "white_point")
    seed_rgb_pl = stable_int_seed(scene_seed, "rgb_point")
    seed_multi_pl = stable_int_seed(scene_seed, "multi_point")
    seed_area = stable_int_seed(scene_seed, "area")
    seed_combined = stable_int_seed(scene_seed, "combined")
    res_dir = f"{args.output_dir}/{file_path.split('/')[-1].split('.')[0]}"
    os.makedirs(res_dir, exist_ok=True)

    #* 1.2 prepare the cameras
    # Check if cameras.json exists in train and test folders
    train_cam_path = os.path.join(res_dir, 'train', 'cameras.json')
    test_cam_path = os.path.join(res_dir, 'test', 'cameras.json')

    def get_scene_bbox_world():
        bbox_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        bbox_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                continue
            for corner in obj.bound_box:
                corner_world = obj.matrix_world @ Vector(corner)
                corner_world_np = np.array(corner_world[:], dtype=np.float32)
                bbox_min = np.minimum(bbox_min, corner_world_np)
                bbox_max = np.maximum(bbox_max, corner_world_np)

        # Fallback to origin if no mesh is found.
        if np.any(np.isinf(bbox_min)) or np.any(np.isinf(bbox_max)):
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        return bbox_min, bbox_max

    # Use the *actual* bbox after scaling+recenter to define a robust scene radius.
    scene_bbox_min, scene_bbox_max = get_scene_bbox_world()
    scene_center = (scene_bbox_min + scene_bbox_max) * 0.5
    scene_radius = float(0.5 * np.linalg.norm(scene_bbox_max - scene_bbox_min))  # half diagonal
    scene_radius = max(scene_radius, 1e-4)

    def get_scene_vertices_world(max_vertices=20000):
        points = []
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        per_object_budget = max(64, max_vertices // max(1, len(mesh_objects)))
        for obj in mesh_objects:
            vertices = obj.data.vertices
            stride = max(1, int(math.ceil(len(vertices) / per_object_budget)))
            for vertex_idx in range(0, len(vertices), stride):
                point_world = obj.matrix_world @ vertices[vertex_idx].co
                points.append([point_world.x, point_world.y, point_world.z])
        if not points:
            raise RuntimeError("Imported scene has no mesh vertices")
        points = np.asarray(points, dtype=np.float64)
        if len(points) > max_vertices:
            indices = np.linspace(0, len(points) - 1, max_vertices, dtype=np.int64)
            points = points[indices]
        return points

    scene_vertices_world = get_scene_vertices_world()

    normalization_info = {
        "scale": scale,
        "offset": array2list(offset),
        "sampled_object_radius": sampled_object_radius,
        "post_normalize_scale": post_normalize_scale,
        "recenter_offset": array2list(recenter_offset),
        "scene_bbox_min": array2list(scene_bbox_min),
        "scene_bbox_max": array2list(scene_bbox_max),
        "scene_center": array2list(scene_center),
        "scene_radius": scene_radius,
    }

    def normalize_vector(vector):
        vector = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            raise ValueError("Cannot normalize a near-zero vector")
        return vector / norm

    def camera_basis(eye, target):
        camera_back = normalize_vector(np.asarray(eye) - np.asarray(target))
        reference_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(reference_up, camera_back))) > 0.98:
            reference_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        camera_right = normalize_vector(np.cross(reference_up, camera_back))
        camera_up = normalize_vector(np.cross(camera_back, camera_right))
        return camera_right, camera_up, camera_back

    def look_at_c2w_with_roll(eye, target, roll_deg):
        camera_right, camera_up, camera_back = camera_basis(eye, target)
        roll = math.radians(roll_deg)
        rolled_right = math.cos(roll) * camera_right + math.sin(roll) * camera_up
        rolled_up = -math.sin(roll) * camera_right + math.cos(roll) * camera_up
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = rolled_right
        c2w[:3, 1] = rolled_up
        c2w[:3, 2] = camera_back
        c2w[:3, 3] = np.asarray(eye, dtype=np.float64)
        return c2w

    def sample_bounded_aim(rng, split):
        sigma = float(profile_config.get(f"aim_sigma_{split}_deg", profile_config["aim_sigma_deg"]))
        clip = float(profile_config.get(f"aim_clip_{split}_deg", profile_config["aim_clip_deg"]))
        yaw = rng.gauss(0.0, sigma)
        pitch = rng.gauss(0.0, sigma)
        magnitude = math.hypot(yaw, pitch)
        if magnitude > clip:
            scale_to_clip = clip / magnitude
            yaw *= scale_to_clip
            pitch *= scale_to_clip
        return yaw, pitch

    scene_fov = scene_rng.uniform(*profile_config["fov_range"])

    def generate_camera_series(num_views, split):
        rng = random.Random(stable_int_seed(scene_seed, split, "camera_series"))
        if num_views <= 0:
            return []

        rho_min, rho_max = profile_config[f"rho_{split}"]
        base_rho = rng.uniform(rho_min, rho_max)
        rho_variation_fraction = float(profile_config["rho_variation_fraction"])
        extent_min, extent_max = profile_config[f"target_extent_{split}"]
        extent_seed_split = "shared" if selected_profile == "stanford" else split
        extent_rng = random.Random(
            stable_int_seed(scene_seed, extent_seed_split, "target_extent")
        )
        base_target_extent = extent_rng.uniform(extent_min, extent_max)
        theta_center = rng.uniform(*profile_config["theta_center_range"])
        theta_amplitude = rng.uniform(*profile_config["theta_amplitude_range"])
        azimuth_span = rng.uniform(*profile_config["azimuth_span_range"])
        azimuth_center = rng.uniform(-180.0, 180.0)
        theta_phase = rng.uniform(0.0, 2.0 * math.pi)
        radius_phase = rng.uniform(0.0, 2.0 * math.pi)
        entries = []
        half_fov_tan = math.tan(math.radians(scene_fov) * 0.5)

        def build_pose(camera_direction, distance, yaw_deg, pitch_deg, roll_deg):
            eye = np.asarray(scene_center, dtype=np.float64) + camera_direction * distance
            camera_right, camera_up, _camera_back = camera_basis(eye, scene_center)
            target = (
                np.asarray(scene_center, dtype=np.float64)
                + distance * math.tan(math.radians(yaw_deg)) * camera_right
                + distance * math.tan(math.radians(pitch_deg)) * camera_up
            )
            return eye, target, look_at_c2w_with_roll(eye, target, roll_deg)

        def projected_mesh_bounds(c2w):
            w2c = np.linalg.inv(c2w)
            camera_points = (
                w2c[:3, :3] @ scene_vertices_world.T + w2c[:3, 3:4]
            ).T
            depth = -camera_points[:, 2]
            valid = depth > 1e-5
            if not np.any(valid):
                return 0.0, 0.0, float("inf")
            normalized_x = camera_points[valid, 0] / (depth[valid] * half_fov_tan)
            normalized_y = camera_points[valid, 1] / (depth[valid] * half_fov_tan)
            bbox_width = float((normalized_x.max() - normalized_x.min()) * 0.5)
            bbox_height = float((normalized_y.max() - normalized_y.min()) * 0.5)
            max_abs_coordinate = float(
                max(np.max(np.abs(normalized_x)), np.max(np.abs(normalized_y)))
            )
            return bbox_width, bbox_height, max_abs_coordinate

        for eye_idx in range(num_views):
            fraction = 0.5 if num_views == 1 else eye_idx / (num_views - 1)
            phi_deg = azimuth_center + azimuth_span * (fraction - 0.5)
            theta_deg = theta_center + theta_amplitude * math.sin(
                2.0 * math.pi * fraction + theta_phase
            )
            theta_deg = min(100.0, max(8.0, theta_deg))

            yaw_deg, pitch_deg = sample_bounded_aim(rng, split)
            center_offset = max(
                abs(math.tan(math.radians(yaw_deg))) / half_fov_tan,
                abs(math.tan(math.radians(pitch_deg))) / half_fov_tan,
            )
            max_center_offset = max(0.05, args.frame_margin - 0.18)
            if center_offset > max_center_offset:
                angular_scale = max_center_offset / center_offset
                yaw_deg *= angular_scale
                pitch_deg *= angular_scale
                center_offset = max_center_offset

            rho_wave = math.sin(2.0 * math.pi * fraction + radius_phase)
            requested_rho = base_rho * (1.0 + rho_variation_fraction * rho_wave)
            requested_rho = min(rho_max, max(rho_min, requested_rho))
            safe_rho = min(requested_rho, max(0.18, args.frame_margin - center_offset))
            distance = scene_radius / max(safe_rho * half_fov_tan, 1e-6)
            distance = max(distance, args.min_camera_clearance * scene_radius)

            theta = math.radians(theta_deg)
            phi = math.radians(phi_deg)
            camera_direction = normalize_vector(np.array(
                [
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                ],
                dtype=np.float64,
            ))
            roll_deg = max(
                -profile_config["roll_clip_deg"],
                min(
                    profile_config["roll_clip_deg"],
                    rng.gauss(0.0, profile_config["roll_sigma_deg"]),
                ),
            )
            requested_extent = base_target_extent * (
                1.0 + rho_variation_fraction * rho_wave
            )
            requested_extent = min(extent_max, max(extent_min, requested_extent))
            safe_target_extent = min(
                requested_extent,
                max(0.15, args.frame_margin - center_offset),
            )

            # Fit distance to the actual projected mesh rather than assuming a
            # bounding-sphere radius predicts silhouette size for every asset.
            for _iteration in range(4):
                eye, target, c2w = build_pose(
                    camera_direction, distance, yaw_deg, pitch_deg, roll_deg
                )
                predicted_width, predicted_height, _max_abs = projected_mesh_bounds(c2w)
                current_extent = max(predicted_width, predicted_height)
                if current_extent <= 1e-6:
                    break
                distance *= current_extent / safe_target_extent
                distance = max(distance, args.min_camera_clearance * scene_radius)

            eye, target, c2w = build_pose(
                camera_direction, distance, yaw_deg, pitch_deg, roll_deg
            )
            predicted_width, predicted_height, max_abs_coordinate = projected_mesh_bounds(c2w)
            if np.isfinite(max_abs_coordinate) and max_abs_coordinate > args.frame_margin:
                distance *= max_abs_coordinate / args.frame_margin
                eye, target, c2w = build_pose(
                    camera_direction, distance, yaw_deg, pitch_deg, roll_deg
                )
                predicted_width, predicted_height, max_abs_coordinate = projected_mesh_bounds(c2w)

            safe_rho = scene_radius / max(distance * half_fov_tan, 1e-6)
            aim_deg = math.degrees(
                math.acos(
                    np.clip(
                        np.dot(normalize_vector(target - eye), normalize_vector(scene_center - eye)),
                        -1.0,
                        1.0,
                    )
                )
            )
            entries.append(
                {
                    "eye_idx": eye_idx,
                    "c2w": c2w.tolist(),
                    "fov": scene_fov,
                    "profile": selected_profile,
                    "rho": float(safe_rho),
                    "requested_rho": float(requested_rho),
                    "requested_max_extent": float(requested_extent),
                    "safe_target_max_extent": float(safe_target_extent),
                    "predicted_bbox_width": float(predicted_width),
                    "predicted_bbox_height": float(predicted_height),
                    "predicted_max_abs_coordinate": float(max_abs_coordinate),
                    "aim_deviation_deg": float(aim_deg),
                    "roll_deg": float(roll_deg),
                    "camera_distance": float(distance),
                    "target_position": target.tolist(),
                }
            )
        return entries

    manifest_path = os.path.join(res_dir, "scene_manifest.json")
    existing_manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as file:
            existing_manifest = json.load(file)

    expected_manifest_identity = {
        "generator_version": GENERATOR_VERSION,
        "object_uid": object_uid,
        "scene_seed": scene_seed,
        "camera_profile": selected_profile,
        "num_views": args.num_views,
        "num_test_views": args.num_test_views,
        "render_resolution": args.render_resolution,
        "cycles_samples": args.cycles_samples,
        "save_intrinsics": args.save_intrinsics,
        "lighting_counts": {
            "white_env": args.num_white_envs,
            "env": args.num_env_lights,
            "white_point": args.num_white_pls,
            "rgb_point": args.num_rgb_pls,
            "multi_point": args.num_multi_pls,
            "area": args.num_area_lights,
            "combined": args.num_combined_lights,
        },
    }
    manifest_matches = existing_manifest is not None and all(
        existing_manifest.get(key) == value
        for key, value in expected_manifest_identity.items()
    )

    has_camera_files = os.path.isfile(train_cam_path) or os.path.isfile(test_cam_path)
    has_legacy_content = any(
        name not in {"scene_manifest.json", "scene_manifest.json.tmp"}
        for name in os.listdir(res_dir)
    )
    if (has_camera_files or has_legacy_content) and not manifest_matches:
        raise RuntimeError(
            f"Refusing to mix V2 cameras with incompatible existing output in {res_dir}. "
            "Use a new output directory or remove this single object directory explicitly."
        )

    loaded_existing_cameras = False
    if manifest_matches and os.path.isfile(train_cam_path) and os.path.isfile(test_cam_path):
        with open(train_cam_path, "r") as file:
            train_cams_data = json.load(file)
        with open(test_cam_path, "r") as file:
            test_cams_data = json.load(file)
        if len(train_cams_data) == args.num_views and len(test_cams_data) == args.num_test_views:
            loaded_existing_cameras = True
            print(f"Loading deterministic V2 cameras from {res_dir}")
        else:
            raise RuntimeError(f"Camera count does not match the manifest in {res_dir}")
    else:
        train_cams_data = generate_camera_series(args.num_views, "train")
        test_cams_data = generate_camera_series(args.num_test_views, "test")

    os.makedirs(os.path.dirname(train_cam_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_cam_path), exist_ok=True)
    atomic_write_json(train_cam_path, train_cams_data)
    atomic_write_json(test_cam_path, test_cams_data)
    cameras = [
        (entry["eye_idx"], Matrix(entry["c2w"]), entry["fov"])
        for entry in train_cams_data
    ]
    cameras_test = [
        (entry["eye_idx"], Matrix(entry["c2w"]), entry["fov"])
        for entry in test_cams_data
    ]

    normalization_info.update(
        {
            "scene_fov": scene_fov,
            "camera_profile": selected_profile,
            "rho_train": list(profile_config["rho_train"]),
            "rho_test": list(profile_config["rho_test"]),
        }
    )
    atomic_write_json(os.path.join(res_dir, "normalize.json"), normalization_info)

    max_combined_stages = 1 + min(2, args.max_pl_num) + 1
    expected_lighting_names = (
        [f"white_env_{idx}" for idx in range(args.num_white_envs)]
        + [f"white_pl_{idx}" for idx in range(args.num_white_pls)]
        + [f"rgb_pl_{idx}" for idx in range(args.num_rgb_pls)]
        + [f"multi_pl_{idx}" for idx in range(args.num_multi_pls)]
        + [f"env_{idx}" for idx in range(args.num_env_lights)]
        + [f"area_{idx}" for idx in range(args.num_area_lights)]
        + [
            f"combined_{idx}"
            for idx in range(min(args.num_combined_lights, max_combined_stages))
        ]
    )
    if not expected_lighting_names:
        raise ValueError("At least one lighting condition must be requested")

    manifest = {
        **expected_manifest_identity,
        "status": "rendering",
        "model_path": file_path,
        "profile_config": profile_config,
        "photo_config": photo_config,
        "normalization": normalization_info,
        "render_settings": {
            "resolution": args.render_resolution,
            "cycles_samples": args.cycles_samples,
            "use_denoising": args.use_denoising,
            "view_transform": "Standard",
            "look": "Medium High Contrast",
            "color_depth": 8,
        },
        "expected_lighting_names": expected_lighting_names,
    }
    atomic_write_json(manifest_path, manifest)

    def hints_complete(view_path, camera_list):
        return all(
            os.path.isfile(os.path.join(view_path, "depth", f"depth_{eye_idx}.exr"))
            and os.path.isfile(os.path.join(view_path, "normal", f"normal_cam_{eye_idx}.exr"))
            and os.path.isfile(os.path.join(view_path, "albedo", f"albedo_cam_{eye_idx}.png"))
            for eye_idx, _c2w, _fov in camera_list
        )

    def render_intrinsic_hints(view_path, camera_list):
        if hints_complete(view_path, camera_list):
            return
        os.makedirs(view_path, exist_ok=True)
        for eye_idx, c2w, fov in camera_list:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            with stdout_redirected():
                render_depth_map(view_path, file_prefix=f"depth_{eye_idx}")
                render_normal_map(view_path)
                render_albedo_map(view_path)
            depth_folder = os.path.join(view_path, "depth")
            normal_folder = os.path.join(view_path, "normal")
            albedo_folder = os.path.join(view_path, "albedo")
            os.makedirs(depth_folder, exist_ok=True)
            os.makedirs(normal_folder, exist_ok=True)
            os.makedirs(albedo_folder, exist_ok=True)

            depth_source = os.path.join(view_path, f"depth_{eye_idx}0001.exr")
            normal_source = os.path.join(view_path, "normal0001.exr")
            albedo_source = os.path.join(view_path, "albedo0001.png")
            shutil.move(depth_source, os.path.join(depth_folder, f"depth_{eye_idx}.exr"))
            transform_normals_to_camera_space(
                normal_source,
                np.asarray(c2w, dtype=np.float64),
                os.path.join(normal_folder, f"normal_cam_{eye_idx}.exr"),
            )
            shutil.move(albedo_source, os.path.join(albedo_folder, f"albedo_cam_{eye_idx}.png"))
            os.remove(normal_source)
            for filename in os.listdir(view_path):
                if filename.startswith("rgb_for_"):
                    os.remove(os.path.join(view_path, filename))
            bpy.data.objects.remove(camera, do_unlink=True)

    if args.save_intrinsics:
        render_intrinsic_hints(os.path.join(res_dir, "train"), cameras)
        render_intrinsic_hints(os.path.join(res_dir, "test"), cameras_test)
    def is_folder_populated(path, num_expected):
        if not os.path.isdir(path):
            return False
        return all(
            os.path.isfile(os.path.join(path, f"gt_{idx}.png"))
            and os.path.getsize(os.path.join(path, f"gt_{idx}.png")) > 0
            for idx in range(num_expected)
        )

    #* 2.1 render the white env lighting first
    for env_idx in range(args.num_white_envs):
        train_env_path = f'{res_dir}/train/white_env_{env_idx}'
        test_env_path = f'{res_dir}/test/white_env_{env_idx}'

        if is_folder_populated(train_env_path, len(cameras)) and is_folder_populated(test_env_path, len(cameras_test)):
            print(f"Skipping existing light: white_env_{env_idx}")
            continue

        # Use the white environment map we created
        env_map_path = f'{args.white_env_map_dir_path}/white_env_8k.exr'
        rotation_euler = [0, 0, random.uniform(-math.pi, math.pi)]
        strength = random.uniform(0.75, 1.25)
        set_env_light(env_map_path, rotation_euler=rotation_euler, strength=strength)

        for eye_idx, c2w, fov in cameras:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/train'
            env_path = f'{view_path}/white_env_{env_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        #* render the test views for white env lighting
        for eye_idx, c2w, fov in cameras_test:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/test'
            env_path = f'{view_path}/white_env_{env_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        # save the env map
        light_info = {
            'env_map': 'white_env_8k.exr',
            'rotation_euler': rotation_euler,
            'strength': strength,
        }
        atomic_write_json(os.path.join(train_env_path, 'white_env.json'), light_info)
        atomic_write_json(os.path.join(test_env_path, 'white_env.json'), light_info)

    light_min_dist = 6.0 * scene_radius
    light_max_dist = 14.0 * scene_radius
    # Keep irradiance approximately invariant to residual normalization-scale
    # differences between assets.
    light_power_scale = (scene_radius / 0.5) ** 2

    #* 2.2 render the white point lighting
    white_pls = gen_random_pts_around_origin(
        seed=seed_white_pl,
        N=args.num_white_pls,
        min_dist_to_origin=light_min_dist,
        max_dist_to_origin=light_max_dist,
        min_theta_in_degree=0,
        max_theta_in_degree=85
    )
    for white_pl_idx in range(args.num_white_pls):
        train_env_path = f'{res_dir}/train/white_pl_{white_pl_idx}'
        test_env_path = f'{res_dir}/test/white_pl_{white_pl_idx}'

        if is_folder_populated(train_env_path, len(cameras)) and is_folder_populated(test_env_path, len(cameras_test)):
            print(f"Skipping existing light: white_pl_{white_pl_idx}")
            continue

        pl = white_pls[white_pl_idx]
        power = random.uniform(500, 1500) * light_power_scale
        _point_light = create_point_light(pl, power)
        set_neutral_world_ambient()

        for eye_idx, c2w, fov in cameras:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/train'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            env_path = f'{view_path}/white_pl_{white_pl_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        #* render the test views for white point lighting
        for eye_idx, c2w, fov in cameras_test:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/test'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            env_path = f'{view_path}/white_pl_{white_pl_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        # save the point light info
        light_info = {
            'pos': array2list(pl),
            'power': power,
        }
        atomic_write_json(os.path.join(train_env_path, 'white_pl.json'), light_info)
        atomic_write_json(os.path.join(test_env_path, 'white_pl.json'), light_info)

    #* 2.3 render the RGB point lighting
    rgb_pls = gen_random_pts_around_origin(
        seed=seed_rgb_pl,
        N=args.num_rgb_pls,
        min_dist_to_origin=light_min_dist,
        max_dist_to_origin=light_max_dist,
        min_theta_in_degree=0,
        max_theta_in_degree=60
    )
    for rgb_pl_idx in range(args.num_rgb_pls):
        train_env_path = f'{res_dir}/train/rgb_pl_{rgb_pl_idx}'
        test_env_path = f'{res_dir}/test/rgb_pl_{rgb_pl_idx}'

        if is_folder_populated(train_env_path, len(cameras)) and is_folder_populated(test_env_path, len(cameras_test)):
            print(f"Skipping existing light: rgb_pl_{rgb_pl_idx}")
            continue

        pl = rgb_pls[rgb_pl_idx]
        power = random.uniform(700, 1400) * light_power_scale
        rgb = [random.uniform(0.55, 1.0) for _ in range(3)]
        color_max = max(rgb)
        rgb = [channel / color_max for channel in rgb]
        create_point_light(pl, power, rgb=rgb)
        set_neutral_world_ambient()

        for eye_idx, c2w, fov in cameras:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/train'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            env_path = f'{view_path}/rgb_pl_{rgb_pl_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        #* render the test views for RGB point lighting
        for eye_idx, c2w, fov in cameras_test:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/test'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            env_path = f'{view_path}/rgb_pl_{rgb_pl_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        # save the RGB point light info
        light_info = {
            'pos': array2list(pl),
            'power': power,
            'color': rgb,
        }
        atomic_write_json(os.path.join(train_env_path, 'rgb_pl.json'), light_info)
        atomic_write_json(os.path.join(test_env_path, 'rgb_pl.json'), light_info)

    #* 2.4 render the multi point lighting
    multi_pls = gen_random_pts_around_origin(
        seed=seed_multi_pl,
        N=args.num_multi_pls * args.max_pl_num,
        min_dist_to_origin=light_min_dist,
        max_dist_to_origin=light_max_dist,
        min_theta_in_degree=0,
        max_theta_in_degree=85
    )

    for multi_pl_idx in range(args.num_multi_pls):
        train_env_path = f'{res_dir}/train/multi_pl_{multi_pl_idx}'
        test_env_path = f'{res_dir}/test/multi_pl_{multi_pl_idx}'

        if is_folder_populated(train_env_path, len(cameras)) and is_folder_populated(test_env_path, len(cameras_test)):
            print(f"Skipping existing light: multi_pl_{multi_pl_idx}")
            continue

        pls = multi_pls[multi_pl_idx * args.max_pl_num: (multi_pl_idx + 1) * args.max_pl_num]
        powers = [
            random.uniform(400, 1200) * light_power_scale
            for _ in range(args.max_pl_num)
        ]
        colors = []
        for pl_idx in range(args.max_pl_num):
            if random.random() < 0.5:
                rgb = [1.0, 1.0, 1.0]  # white
            else:
                rgb = [random.uniform(0.55, 1.0) for _ in range(3)]
                color_max = max(rgb)
                rgb = [channel / color_max for channel in rgb]
            colors.append(rgb)
            create_point_light(pls[pl_idx], powers[pl_idx], rgb=rgb, keep_other_lights=pl_idx > 0)
        set_neutral_world_ambient()

        for eye_idx, c2w, fov in cameras:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/train'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            env_path = f'{view_path}/multi_pl_{multi_pl_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        #* render the test views for multi point lighting
        for eye_idx, c2w, fov in cameras_test:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/test'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            env_path = f'{view_path}/multi_pl_{multi_pl_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        # save the multi point light info
        light_info = {
            'pos': mat2list(pls),
            'power': powers,
            'color': colors,
        }
        atomic_write_json(os.path.join(train_env_path, 'multi_pl.json'), light_info)
        atomic_write_json(os.path.join(test_env_path, 'multi_pl.json'), light_info)

    #* 2.5 render the colored env lighting
    for env_map_idx in range(args.num_env_lights):
        train_env_path = f'{res_dir}/train/env_{env_map_idx}'
        test_env_path = f'{res_dir}/test/env_{env_map_idx}'

        if is_folder_populated(train_env_path, len(cameras)) and is_folder_populated(test_env_path, len(cameras_test)):
            print(f"Skipping existing light: env_{env_map_idx}")
            continue

        env_map = random.choice(env_map_list)
        env_map_path = f'{args.env_map_dir_path}/{env_map}_8k.exr'
        rotation_euler = [0, 0, random.uniform(-math.pi, math.pi)]
        strength = random.uniform(0.65, 1.35)
        set_env_light(env_map_path, rotation_euler=rotation_euler, strength=strength)

        for eye_idx, c2w, fov in cameras:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/train'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            env_path = f'{view_path}/env_{env_map_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        #* render the test views for colored env lighting
        for eye_idx, c2w, fov in cameras_test:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/test'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            env_path = f'{view_path}/env_{env_map_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        # save the env map
        light_info = {
            'env_map': env_map,
            'rotation_euler': rotation_euler,
            'strength': strength,
        }
        atomic_write_json(os.path.join(train_env_path, 'env.json'), light_info)
        atomic_write_json(os.path.join(test_env_path, 'env.json'), light_info)

    #* 2.6 render the area lighting
    area_light_positions = gen_random_pts_around_origin(
        seed=seed_area,
        N=args.num_area_lights,
        min_dist_to_origin=light_min_dist,
        max_dist_to_origin=light_max_dist,
        min_theta_in_degree=0,
        max_theta_in_degree=85
    )
    for area_light_idx in range(args.num_area_lights):
        train_env_path = f'{res_dir}/train/area_{area_light_idx}'
        test_env_path = f'{res_dir}/test/area_{area_light_idx}'

        if is_folder_populated(train_env_path, len(cameras)) and is_folder_populated(test_env_path, len(cameras_test)):
            print(f"Skipping existing light: area_{area_light_idx}")
            continue

        area_light_pos = area_light_positions[area_light_idx]
        area_light_power = random.uniform(500, 1400) * light_power_scale
        area_light_size = random.uniform(1.5, 5.0) * scene_radius
        if random.random() < 0.75:
            color = [1.0, 1.0, 1.0]  # white
        else:
            color = [random.uniform(0.6, 1.0) for _ in range(3)]
            color_max = max(color)
            color = [channel / color_max for channel in color]

        _area_light = create_area_light(area_light_pos, area_light_power, area_light_size, color=color)
        set_neutral_world_ambient()

        for eye_idx, c2w, fov in cameras:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/train'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            env_path = f'{view_path}/area_{area_light_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        #* render the test views for area lighting
        for eye_idx, c2w, fov in cameras_test:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/test'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            env_path = f'{view_path}/area_{area_light_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        # save the area light info
        light_info = {
            'pos': array2list(area_light_pos),
            'power': area_light_power,
            'size': area_light_size,
            'color': color,
        }
        atomic_write_json(os.path.join(train_env_path, 'area.json'), light_info)
        atomic_write_json(os.path.join(test_env_path, 'area.json'), light_info)

    #* 2.7 render the combined lighting (progressive: env -> +point1 -> +point2 -> +area)
    # Generate positions for point lights and area light
    num_point_lights = min(2, args.max_pl_num)  # Use up to 2 point lights
    combined_pls = gen_random_pts_around_origin(
        seed=seed_combined,
        N=num_point_lights,
        min_dist_to_origin=light_min_dist,
        max_dist_to_origin=light_max_dist,
        min_theta_in_degree=0,
        max_theta_in_degree=85
    )
    area_light_positions = gen_random_pts_around_origin(
        seed=seed_combined + 100 if seed_combined is not None else None,
        N=1,
        min_dist_to_origin=light_min_dist,
        max_dist_to_origin=light_max_dist,
        min_theta_in_degree=0,
        max_theta_in_degree=85
    )

    if args.num_combined_lights > 0:
        # Setup: Choose env map parameters (shared across all progressive stages)
        env_map = random.choice(env_map_list)
        env_map_path = f'{args.env_map_dir_path}/{env_map}_8k.exr'
        rotation_euler = [0, 0, random.uniform(-math.pi, math.pi)]
        strength = random.uniform(0.65, 1.25)

        # Generate light parameters
        point_powers = [
            random.uniform(300, 1200) * light_power_scale
            for _ in range(num_point_lights)
        ]
        point_colors = [[1.0, 1.0, 1.0] for _ in range(num_point_lights)]

        area_light_pos = area_light_positions[0]
        area_light_power = random.uniform(400, 1200) * light_power_scale
        area_light_size = random.uniform(1.5, 5.0) * scene_radius
        area_light_color = [1.0, 1.0, 1.0]  # white area light

        # Progressive rendering stages
        # Stage 0: env map only
        # Stage 1: env + 1st point light
        # Stage 2: env + 1st point + 2nd point light
        # Stage 3: env + 1st point + 2nd point + area light
        max_stages = 1 + num_point_lights + 1  # env + points + area
        num_stages = min(args.num_combined_lights, max_stages)

        for stage_idx in range(num_stages):
            train_env_path = f'{res_dir}/train/combined_{stage_idx}'
            test_env_path = f'{res_dir}/test/combined_{stage_idx}'

            if is_folder_populated(train_env_path, len(cameras)) and is_folder_populated(test_env_path, len(cameras_test)):
                print(f"Skipping existing light: combined_{stage_idx}")
                continue

            # Stage 0: Set env light
            if stage_idx == 0:
                set_env_light(env_map_path, rotation_euler=rotation_euler, strength=strength)
                light_info = {
                    'stage': 0,
                    'description': 'env_only',
                    'env_map': env_map,
                    'rotation_euler': rotation_euler,
                    'strength': strength,
                }

            # Stage 1: Add 1st point light
            elif stage_idx == 1:
                set_env_light(env_map_path, rotation_euler=rotation_euler, strength=strength)
                create_point_light(combined_pls[0], point_powers[0], rgb=point_colors[0], keep_other_lights=True)
                light_info = {
                    'stage': 1,
                    'description': 'env + 1 point light',
                    'env_map': env_map,
                    'has_env_light': True,
                    'rotation_euler': rotation_euler,
                    'strength': strength,
                    'point_lights': {
                        'pos': [array2list(combined_pls[0])],
                        'power': [point_powers[0]],
                        'color': [point_colors[0]],
                    }
                }

            # Stage 2: Add 2nd point light (if available)
            elif stage_idx == 2 and num_point_lights >= 2:
                set_env_light(env_map_path, rotation_euler=rotation_euler, strength=strength)
                create_point_light(combined_pls[0], point_powers[0], rgb=point_colors[0], keep_other_lights=True)
                create_point_light(combined_pls[1], point_powers[1], rgb=point_colors[1], keep_other_lights=True)
                light_info = {
                    'stage': 2,
                    'description': 'env + 2 point lights',
                    'env_map': env_map,
                    'has_env_light': True,
                    'rotation_euler': rotation_euler,
                    'strength': strength,
                    'point_lights': {
                        'pos': [array2list(combined_pls[0]), array2list(combined_pls[1])],
                        'power': [point_powers[0], point_powers[1]],
                        'color': [point_colors[0], point_colors[1]],
                    }
                }

            # Stage 3+: Add area light
            elif stage_idx >= 3 or (stage_idx == 2 and num_point_lights < 2):
                set_env_light(env_map_path, rotation_euler=rotation_euler, strength=strength)
                # Add all available point lights
                for pl_idx in range(num_point_lights):
                    create_point_light(combined_pls[pl_idx], point_powers[pl_idx], rgb=point_colors[pl_idx], keep_other_lights=True)
                # Add area light
                create_area_light(area_light_pos, area_light_power, area_light_size, color=area_light_color, keep_other_lights=True)

                point_lights_data = {
                    'pos': [array2list(combined_pls[i]) for i in range(num_point_lights)],
                    'power': point_powers[:num_point_lights],
                    'color': point_colors[:num_point_lights],
                } if num_point_lights > 0 else None

                light_info = {
                    'stage': 3,
                    'description': f'env + {num_point_lights} point lights + area light',
                    'env_map': env_map,
                    'rotation_euler': rotation_euler,
                    'has_env_light': True,
                    'strength': strength,
                    'point_lights': point_lights_data,
                    'area_light': {
                        'pos': array2list(area_light_pos),
                        'power': area_light_power,
                        'size': area_light_size,
                        'color': area_light_color,
                    }
                }

            # Render train views
            for eye_idx, c2w, fov in cameras:
                camera = create_camera(c2w, fov)
                bpy.context.scene.camera = camera
                view_path = f'{res_dir}/train'
                if not os.path.exists(view_path):
                    os.makedirs(view_path)

                env_path = f'{view_path}/combined_{stage_idx}'
                os.makedirs(env_path, exist_ok=True)
                with stdout_redirected():
                    render_rgb_and_hint(f'{env_path}', eye_idx)

                bpy.data.objects.remove(camera, do_unlink=True)

            # Save light info for train
            train_env_path = f'{res_dir}/train/combined_{stage_idx}'
            atomic_write_json(os.path.join(train_env_path, 'combined.json'), light_info)

            # Render test views
            for eye_idx, c2w, fov in cameras_test:
                camera = create_camera(c2w, fov)
                bpy.context.scene.camera = camera
                view_path = f'{res_dir}/test'
                if not os.path.exists(view_path):
                    os.makedirs(view_path)

                env_path = f'{view_path}/combined_{stage_idx}'
                os.makedirs(env_path, exist_ok=True)
                with stdout_redirected():
                    render_rgb_and_hint(f'{env_path}', eye_idx)

                bpy.data.objects.remove(camera, do_unlink=True)

            # Save light info for test
            test_env_path = f'{res_dir}/test/combined_{stage_idx}'
            atomic_write_json(os.path.join(test_env_path, 'combined.json'), light_info)

    def quantiles(values):
        if not values:
            return []
        return [
            float(value)
            for value in np.quantile(np.asarray(values, dtype=np.float64), [0.05, 0.5, 0.95])
        ]

    def validate_rendered_scene():
        report = {
            "missing_files": [],
            "decode_errors": [],
            "empty_frames": 0,
            "boundary_touch_frames": 0,
            "mask_area_invalid_frames": 0,
            "photometric_invalid_frames": 0,
            "sampled_frames": 0,
            "photometric_sampled_frames": 0,
            "mask_areas": [],
            "bbox_widths": [],
            "bbox_heights": [],
            "foreground_luma": [],
        }

        for split, expected_count in (("train", args.num_views), ("test", args.num_test_views)):
            camera_path = os.path.join(res_dir, split, "cameras.json")
            if not os.path.isfile(camera_path):
                report["missing_files"].append(camera_path)
            for light_name in expected_lighting_names:
                light_path = os.path.join(res_dir, split, light_name)
                for eye_idx in range(expected_count):
                    frame_path = os.path.join(light_path, f"gt_{eye_idx}.png")
                    if not os.path.isfile(frame_path) or os.path.getsize(frame_path) == 0:
                        report["missing_files"].append(frame_path)

        # Geometry is independent of illumination. Decode one complete lighting
        # condition for all cameras, and use all conditions only for exact-file checks.
        reference_light = expected_lighting_names[0]
        for split, expected_count in (("train", args.num_views), ("test", args.num_test_views)):
            for eye_idx in range(expected_count):
                frame_path = os.path.join(res_dir, split, reference_light, f"gt_{eye_idx}.png")
                if not os.path.isfile(frame_path):
                    continue
                try:
                    image = imageio.v3.imread(frame_path)
                except Exception as error:
                    report["decode_errors"].append({"path": frame_path, "error": str(error)})
                    continue
                report["sampled_frames"] += 1
                max_value = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else 1.0
                rgb = image[..., :3].astype(np.float32) / max_value
                if image.shape[-1] == 4:
                    mask = image[..., 3].astype(np.float32) / max_value > 0.01
                else:
                    mask = np.any(rgb < 0.99, axis=-1)
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    report["empty_frames"] += 1
                    continue

                height, width = mask.shape
                mask_area = float(mask.mean())
                bbox_width = float((xs.max() - xs.min() + 1) / width)
                bbox_height = float((ys.max() - ys.min() + 1) / height)
                touches_boundary = bool(
                    xs.min() == 0 or ys.min() == 0 or xs.max() == width - 1 or ys.max() == height - 1
                )
                foreground_luma = float(
                    (rgb[mask] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).mean()
                )
                report["mask_areas"].append(mask_area)
                report["bbox_widths"].append(bbox_width)
                report["bbox_heights"].append(bbox_height)
                report["foreground_luma"].append(foreground_luma)
                report["photometric_sampled_frames"] += 1
                report["boundary_touch_frames"] += int(touches_boundary)
                report["mask_area_invalid_frames"] += int(
                    mask_area < args.min_mask_area or mask_area > args.max_mask_area
                )
                report["photometric_invalid_frames"] += int(
                    foreground_luma < args.min_foreground_luma
                    or foreground_luma > args.max_foreground_luma
                )

        # Check every other illumination for black/overexposed supervision. Mask
        # geometry is already validated above and is invariant to illumination.
        for light_name in expected_lighting_names[1:]:
            for split, expected_count in (("train", args.num_views), ("test", args.num_test_views)):
                for eye_idx in range(expected_count):
                    frame_path = os.path.join(res_dir, split, light_name, f"gt_{eye_idx}.png")
                    if not os.path.isfile(frame_path):
                        continue
                    try:
                        image = imageio.v3.imread(frame_path)
                    except Exception as error:
                        report["decode_errors"].append({"path": frame_path, "error": str(error)})
                        continue
                    max_value = (
                        float(np.iinfo(image.dtype).max)
                        if np.issubdtype(image.dtype, np.integer)
                        else 1.0
                    )
                    rgb = image[..., :3].astype(np.float32) / max_value
                    if image.shape[-1] == 4:
                        mask = image[..., 3].astype(np.float32) / max_value > 0.01
                    else:
                        mask = np.any(rgb < 0.99, axis=-1)
                    if not np.any(mask):
                        continue
                    foreground_luma = float(
                        (rgb[mask] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).mean()
                    )
                    report["foreground_luma"].append(foreground_luma)
                    report["photometric_sampled_frames"] += 1
                    report["photometric_invalid_frames"] += int(
                        foreground_luma < args.min_foreground_luma
                        or foreground_luma > args.max_foreground_luma
                    )

        sampled = max(1, report["sampled_frames"])
        photometric_sampled = max(1, report["photometric_sampled_frames"])
        report["boundary_touch_fraction"] = report["boundary_touch_frames"] / sampled
        report["geometry_invalid_fraction"] = (
            report["empty_frames"] + report["mask_area_invalid_frames"]
        ) / sampled
        report["photometric_invalid_fraction"] = (
            report["photometric_invalid_frames"] / photometric_sampled
        )
        report["mask_area_q05_median_q95"] = quantiles(report.pop("mask_areas"))
        report["bbox_width_q05_median_q95"] = quantiles(report.pop("bbox_widths"))
        report["bbox_height_q05_median_q95"] = quantiles(report.pop("bbox_heights"))
        report["foreground_luma_q05_median_q95"] = quantiles(report.pop("foreground_luma"))
        report["aim_train_q05_median_q95"] = quantiles(
            [entry.get("aim_deviation_deg", 0.0) for entry in train_cams_data]
        )
        report["aim_test_q05_median_q95"] = quantiles(
            [entry.get("aim_deviation_deg", 0.0) for entry in test_cams_data]
        )
        camera_distance_ratios = [
            entry.get("camera_distance", 0.0) / scene_radius
            for entry in train_cams_data + test_cams_data
        ]
        report["camera_distance_over_radius_q05_median_q95"] = quantiles(
            camera_distance_ratios
        )
        report["minimum_camera_distance_over_radius"] = min(camera_distance_ratios)
        report["camera_clearance_passed"] = bool(
            report["minimum_camera_distance_over_radius"] >= args.min_camera_clearance
        )

        report["passed"] = bool(
            not report["missing_files"]
            and not report["decode_errors"]
            and report["camera_clearance_passed"]
            and report["geometry_invalid_fraction"] == 0.0
            and report["boundary_touch_fraction"] <= args.max_boundary_touch_fraction
            and report["photometric_invalid_fraction"] <= args.max_photometric_invalid_fraction
        )
        return report

    validation_report = validate_rendered_scene()
    atomic_write_json(os.path.join(res_dir, "validation_report.json"), validation_report)
    manifest["status"] = "complete" if validation_report["passed"] else "invalid"
    manifest["validation_summary"] = validation_report
    atomic_write_json(manifest_path, manifest)

    done_path = os.path.join(res_dir, "done.txt")
    invalid_path = os.path.join(res_dir, "invalid.txt")
    for marker_path in (done_path, invalid_path):
        if os.path.isfile(marker_path):
            os.remove(marker_path)
    marker_path = done_path if validation_report["passed"] else invalid_path
    with open(marker_path, "w") as file:
        file.write("complete\n" if validation_report["passed"] else "validation failed\n")
    if not validation_report["passed"]:
        raise RuntimeError(
            f"Rendered scene failed validation; inspect {os.path.join(res_dir, 'validation_report.json')}"
        )


if __name__ == '__main__':
    args: Options = simple_parsing.parse(Options)
    print(args)
    dataset_path = os.path.abspath(args.models_root)
    args.output_dir = os.path.abspath(args.output_dir_override or args.output_dir)
    if args.num_view_groups != 1:
        raise ValueError("Enhanced V2 currently requires num_view_groups=1 for one manifest per object")
    import csv
    index_uid_list = []
    with open(args.csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                index, uid = row
                index_uid_list.append((index.strip(), uid.strip()))
    # Preview
    print(f"Loaded {len(index_uid_list)} entries")
    if not index_uid_list:
        raise ValueError(f"No 'shard,uid' rows found in {args.csv_path}")
    if args.group_start < 0 or args.group_end > len(index_uid_list) or args.group_start >= args.group_end:
        raise ValueError(
            f"Expected 0 <= group_start < group_end <= {len(index_uid_list)}, got "
            f"{args.group_start}:{args.group_end}"
        )

    for i in range(args.group_start, args.group_end):
        index, uid = index_uid_list[i]
        model_path = os.path.join(dataset_path, index, f'{uid}.glb')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model does not exist: {model_path}")
        args.three_d_model_path = model_path
        os.makedirs(args.output_dir, exist_ok=True)
        # Render the model
        print('Rendering model:', uid)
        if uid in error_list:
            print('skipping this model')
            continue
        for j in range(args.num_view_groups):
            # if found a done.txt file, skip this model
            print('rendering group:', j)
            # if os.path.exists(os.path.join(args.output_dir, uid, 'done.txt')):
            #     continue
            render_core(args, j)
            print('render progress:', i, 'of range', args.group_start, '~', args.group_end)
