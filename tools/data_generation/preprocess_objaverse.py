#!/usr/bin/env python3
"""
Preprocessing script to convert Objaverse data format to re10k format.

Migrated from the LumiTokens development repository and adapted for the public
release. The release version preserves all source renders; optional archives
are written only beneath --output-tar.

The script processes Objaverse data where:
- Each object has train/ and test/ folders
- Scenes can be lit by environment lights or point lights:
  - Env lights: env_0/, env_1/, ..., white_env_0/ (lighting from env.json / white_env.json + HDRI)
  - Point lights: rgb_pl_0/, white_pl_0/, ... (lighting from rgb_pl.json / white_pl.json; color (1,1,1) for white_pl)
- Each scene folder contains gt_{idx}.png images
- cameras.json contains camera parameters in Blender convention

Output format matches re10k:
- JSON files with scene_name and frames array (metadata/)
- Each frame has: image_path, fxfycxcy, w2c
- Images in images/{scene_name}/, envmaps in envmaps/{scene_name}/ (env-lit only)
- Point-light rays in point_light_rays/{scene_name}.npy (point-lit only): [N, 10] = intensity(1) + color(3) + ray_o(3) + ray_d(3)
"""

import os
import json
import numpy as np
import argparse
import io
from pathlib import Path
from PIL import Image
import shutil
import torch
import torch.nn.functional as F
import cv2
import tarfile
import subprocess
try:
    import pyexr
    HAS_PYEXR = True
except ImportError:
    HAS_PYEXR = False
    print("Warning: pyexr not available, HDR EXR files cannot be read")


def blender_to_opencv_c2w(c2w_blender):
    """
    Convert Blender c2w matrix to OpenCV c2w matrix.

    Blender convention: +X right, +Y forward, +Z up
    OpenCV convention: +X right, +Y down, +Z forward

    Transformation: Rotate by [1,0,0][0,-1,0][0,0,-1] on rotation part only
    Translation vector remains unchanged.

    Args:
        c2w_blender: 4x4 numpy array in Blender convention

    Returns:
        c2w_opencv: 4x4 numpy array in OpenCV convention
    """
    c2w_opencv = c2w_blender.copy()

    # Transformation matrix for rotation part only
    # [1,  0,  0]
    # [0, -1,  0]
    # [0,  0, -1]
    transform = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    # Apply transformation to rotation part (3x3 top-left)
    # Multiply from the right: R_opencv = R_blender @ transform
    c2w_opencv[:3, :3] = c2w_blender[:3, :3] @ transform

    # Translation (last column, first 3 rows) remains unchanged
    # Already copied, no change needed

    return c2w_opencv


def fov_to_fxfycxcy(fov_degrees, image_width, image_height):
    """
    Convert field of view (FOV) to fxfycxcy intrinsic parameters.

    Args:
        fov_degrees: Field of view in degrees
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        fxfycxcy: [fx, fy, cx, cy] array
    """
    fov_rad = np.radians(fov_degrees)
    fx = fy = (image_width / 2.0) / np.tan(fov_rad / 2.0)
    cx = image_width / 2.0
    cy = image_height / 2.0
    return [fx, fy, cx, cy]


def is_pl_folder(name):
    """Return True if folder name indicates point-light lighting (rgb_pl_* or white_pl_*)."""
    return (name.startswith('rgb_pl_') or name.startswith('white_pl_'))


def is_multi_pl_folder(name):
    """Return True if folder name indicates multi-point-light lighting (multi_pl_*)."""
    return name.startswith('multi_pl_')


def is_area_folder(name):
    """Return True if folder name indicates area light lighting (area_*)."""
    return name.startswith('area_')


def is_combined_folder(name):
    """Return True if folder name indicates combined lighting (combined_*)."""
    return name.startswith('combined_')


def load_point_light_info(pl_folder_path):
    """
    Load point light position, color and power from rgb_pl.json or white_pl.json.
    Returns (pos, color, power) or None if not found.
    For white_pl, color is (1, 1, 1).
    """
    json_names = ("rgb_pl.json", "white_pl.json")
    data = None

    if os.path.isdir(pl_folder_path):
        for name in json_names:
            path = os.path.join(pl_folder_path, name)
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                break
    elif os.path.isfile(pl_folder_path) and pl_folder_path.endswith(".tar"):
        with tarfile.open(pl_folder_path, "r") as tar:
            file_members = [m for m in tar.getmembers() if m.isfile()]
            member_by_name = {m.name: m for m in file_members}
            member_by_base = {os.path.basename(m.name): m for m in file_members}
            for name in json_names:
                member = member_by_name.get(name) or member_by_base.get(name)
                if member is not None:
                    fobj = tar.extractfile(member)
                    if fobj is not None:
                        data = json.load(fobj)
                    break

    if data is not None:
        pos = np.array(data["pos"], dtype=np.float64)
        power = float(data["power"])
        color = np.array(data.get("color", [1.0, 1.0, 1.0]), dtype=np.float64)
        if color.size != 3:
            color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        return pos, color, power
    return None


def uniform_sphere_surface(N, center=(0, 0, 0), radius=0.5):
    """Sample N points uniformly on the surface of a sphere (Fibonacci sphere)."""
    center = np.asarray(center, dtype=np.float64)
    indices = np.arange(N, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    y = 1.0 - (indices / max(N - 1, 1)) * 2.0  # y from 1 to -1
    r_xy = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    theta = phi * indices
    x = np.cos(theta) * r_xy
    z = np.sin(theta) * r_xy
    points = np.stack([x, y, z], axis=1) * radius
    return points + center


# Point light power range; we normalize to [0,1] then apply HDR log. POWER_MIN=10 allows smaller intensities.
POWER_MIN = 10.0
POWER_MAX = 1500.0


def build_point_light_rays_array(pos, color, power, N=8192, scene_sphere_radius=3.0):
    """
    Build ray array for a point light: rays from light origin to uniformly sampled points on sphere.
    Sphere: center (0,0,0), radius scene_sphere_radius. Intensity: power is normalized to [0,1] by
    the known sampling range [POWER_MIN, POWER_MAX], then log-normalized like HDR (log1p(10*x)) so
    intensity lies in [0, log1p(10)] ~ [0, 2.4], matching env map scale.
    Shape: [N, 10] = 1 (intensity) + 3 (color) + 3 (ray_o) + 3 (ray_d), ray_d normalized.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    color = np.asarray(color, dtype=np.float64).reshape(3)
    points = uniform_sphere_surface(N, center=(0, 0, 0), radius=scene_sphere_radius)
    ray_o = np.broadcast_to(pos, (N, 3)).astype(np.float32)
    ray_d = points - pos
    norms = np.linalg.norm(ray_d, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    ray_d = (ray_d / norms).astype(np.float32)
    power_clip = np.clip(power, POWER_MIN, POWER_MAX)
    power_norm = (float(power_clip) - POWER_MIN) / (POWER_MAX - POWER_MIN)
    intensity = np.log1p(10.0 * power_norm)
    intensity = np.full((N, 1), intensity, dtype=np.float32)
    color_broadcast = np.broadcast_to(color.astype(np.float32), (N, 3))
    arr = np.concatenate([intensity, color_broadcast, ray_o, ray_d], axis=1)
    return arr


def load_multi_point_light_info(folder_path):
    """
    Load multi-point-light info from multi_pl.json.
    Returns list of (pos, color, power) tuples, or None if not found.
    """
    path = os.path.join(folder_path, 'multi_pl.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    positions = [np.array(p, dtype=np.float64) for p in data['pos']]
    powers = [float(pw) for pw in data['power']]
    colors = [np.array(c, dtype=np.float64) for c in data['color']]
    return list(zip(positions, colors, powers))


def load_area_light_info(folder_path):
    """
    Load area light info from area.json.
    Returns (pos, color, power, size) or None if not found.
    """
    path = os.path.join(folder_path, 'area.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    pos = np.array(data['pos'], dtype=np.float64)
    power = float(data['power'])
    size = float(data['size'])
    color = np.array(data.get('color', [1.0, 1.0, 1.0]), dtype=np.float64)
    return pos, color, power, size


def load_combined_light_info(folder_path):
    """
    Load combined lighting info from combined.json.
    Handles multiple JSON formats:
      - Format A (with 'stage'): has description, env_map, point_lights dict, area_light dict
      - Format B (flat): has env_map, pos/power/color at top level (single point light)

    Returns dict with keys:
      'env_map': str or None
      'rotation_euler': list or None
      'strength': float
      'point_lights': list of (pos, color, power) or []
      'area_lights': list of (pos, color, power, size) or []
    """
    path = os.path.join(folder_path, 'combined.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)

    result = {
        'env_map': data.get('env_map', None),
        'rotation_euler': data.get('rotation_euler', None),
        'strength': float(data.get('strength', 1.0)),
        'point_lights': [],
        'area_lights': [],
    }

    # Format A: structured point_lights dict
    if 'point_lights' in data:
        pl = data['point_lights']
        for i in range(len(pl['pos'])):
            pos = np.array(pl['pos'][i], dtype=np.float64)
            power = float(pl['power'][i])
            color = np.array(pl['color'][i], dtype=np.float64)
            result['point_lights'].append((pos, color, power))

    # Format B: single point light at top level (pos, power, color but also has env_map)
    elif 'pos' in data and 'power' in data and 'env_map' in data:
        pos = np.array(data['pos'], dtype=np.float64)
        power = float(data['power'])
        color = np.array(data.get('color', [1.0, 1.0, 1.0]), dtype=np.float64)
        result['point_lights'].append((pos, color, power))

    # Area light (Format A only)
    if 'area_light' in data:
        al = data['area_light']
        pos = np.array(al['pos'], dtype=np.float64)
        power = float(al['power'])
        size = float(al['size'])
        color = np.array(al.get('color', [1.0, 1.0, 1.0]), dtype=np.float64)
        result['area_lights'].append((pos, color, power, size))

    return result


def build_area_light_rays_array(pos, color, power, size, N=8192, scene_sphere_radius=3.0):
    """
    Build ray array for an area light.
    Rays originate from random points on a square plane centered at `pos`, with normal facing
    world center (0,0,0), and side length `size`. They point toward random targets on a
    scene sphere of radius `scene_sphere_radius` centered at (0,0,0).
    Shape: [N, 10] = 1 (intensity) + 3 (color) + 3 (ray_o) + 3 (ray_d), ray_d normalized.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    color = np.asarray(color, dtype=np.float64).reshape(3)

    # Build orthonormal basis for the plane: normal faces world center
    normal = -pos / (np.linalg.norm(pos) + 1e-8)  # pointing toward origin
    # Pick an arbitrary "up" that isn't parallel to normal
    up_candidate = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, up_candidate)) > 0.99:
        up_candidate = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(normal, up_candidate)
    tangent /= np.linalg.norm(tangent) + 1e-8
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent) + 1e-8

    # Sample random origins on the plane
    half = size / 2.0
    u = np.random.uniform(-half, half, size=N)
    v = np.random.uniform(-half, half, size=N)
    ray_o = pos[None, :] + u[:, None] * tangent[None, :] + v[:, None] * bitangent[None, :]

    # Sample random target points on scene sphere
    targets = uniform_sphere_surface(N, center=(0, 0, 0), radius=scene_sphere_radius)

    # Direction from origin to target, then normalize
    ray_d = targets - ray_o
    norms = np.linalg.norm(ray_d, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    ray_d = (ray_d / norms).astype(np.float32)
    ray_o = ray_o.astype(np.float32)

    # Normalize power
    power_clip = np.clip(power, POWER_MIN, POWER_MAX)
    power_norm = (float(power_clip) - POWER_MIN) / (POWER_MAX - POWER_MIN)
    intensity = np.log1p(10.0 * power_norm)
    intensity = np.full((N, 1), intensity, dtype=np.float32)
    color_broadcast = np.broadcast_to(color.astype(np.float32), (N, 3))

    arr = np.concatenate([intensity, color_broadcast, ray_o, ray_d], axis=1)
    return arr


def build_multi_source_light_rays(point_lights, area_lights, N=8192, scene_sphere_radius=3.0):
    """
    Build a combined ray array from multiple point lights and/or area lights.
    The total number of rays is N, allocated proportionally to each light's power.
    Shape: [N, 10]
    """
    all_sources = []  # list of (type, data, power)
    for (pos, color, power) in point_lights:
        all_sources.append(('point', (pos, color, power), power))
    for (pos, color, power, size) in area_lights:
        all_sources.append(('area', (pos, color, power, size), power))

    if not all_sources:
        return np.zeros((N, 10), dtype=np.float32)

    # Allocate rays proportional to power
    total_power = sum(s[2] for s in all_sources)
    if total_power < 1e-8:
        # Equal allocation if all powers are zero
        counts = [N // len(all_sources)] * len(all_sources)
    else:
        counts = [max(1, int(round(N * (s[2] / total_power)))) for s in all_sources]
    # Adjust to make total exactly N
    diff = N - sum(counts)
    counts[0] += diff

    arrays = []
    for (src_type, src_data, _), n_rays in zip(all_sources, counts):
        if n_rays <= 0:
            continue
        if src_type == 'point':
            pos, color, power = src_data
            arr = build_point_light_rays_array(pos, color, power, N=n_rays)
        else:
            pos, color, power, size = src_data
            arr = build_area_light_rays_array(pos, color, power, size, N=n_rays, scene_sphere_radius=scene_sphere_radius)
        arrays.append(arr)

    return np.concatenate(arrays, axis=0)[:N]  # ensure exactly N


def check_scene_broken(split_path, object_id):
    """
    Check if a scene is broken (no materials) by checking albedo mask and RGB images.

    Args:
        split_path: Path to the split folder (test/ or train/)
        object_id: Object ID

    Returns:
        bool: True if scene is broken, False otherwise
    """
    albedo_dir = os.path.join(split_path, 'albedo')
    if not os.path.exists(albedo_dir):
        return False  # No albedo folder, cannot check

    # Get first albedo image to check mask
    albedo_files = sorted([f for f in os.listdir(albedo_dir) if f.endswith('.png')])
    if not albedo_files:
        return False  # No albedo images

    first_albedo_path = os.path.join(albedo_dir, albedo_files[0])
    try:
        albedo_img = Image.open(first_albedo_path)
        if albedo_img.mode == 'RGBA':
            albedo_array = np.array(albedo_img)
            alpha_channel = albedo_array[:, :, 3]
            # Mask is where alpha is 0
            mask = (alpha_channel == 0)
            mask_ratio = np.sum(mask) / mask.size

            # If mask takes up less than 1/4 of image, mark as broken
            if mask_ratio < 0.25:
                # Check RGB images inside the mask
                # Find corresponding RGB image (gt_0.png for albedo_cam_0.png)
                cam_idx = albedo_files[0].replace('albedo_cam_', '').replace('.png', '')
                try:
                    cam_idx_int = int(cam_idx)
                    # Look for gt_{cam_idx}.png in any env or point-light folder
                    light_folders = [d for d in os.listdir(split_path)
                                     if os.path.isdir(os.path.join(split_path, d))
                                     and ((d.startswith('env_') or d.startswith('white_env_'))
                                          or is_pl_folder(d))]
                    for light_folder in light_folders:
                        light_path = os.path.join(split_path, light_folder)
                        rgb_path = os.path.join(light_path, f'gt_{cam_idx_int}.png')
                        if os.path.exists(rgb_path):
                            rgb_img = Image.open(rgb_path).convert('RGB')
                            rgb_array = np.array(rgb_img)

                            # Get RGB values inside the mask
                            masked_rgb = rgb_array[mask]
                            if len(masked_rgb) > 0:
                                # Check if average color is all black (smaller than (1,1,1) in 0-255 scale)
                                avg_color = np.mean(masked_rgb, axis=0)
                                if np.all(avg_color < 1.0):
                                    return True  # Scene is broken
                            break
                except (ValueError, FileNotFoundError):
                    pass
        else:
            # If albedo doesn't have alpha channel, check if it's all black or very dark
            albedo_array = np.array(albedo_img.convert('RGB'))
            avg_color = np.mean(albedo_array, axis=(0, 1))
            if np.all(avg_color < 1.0):
                return True  # Scene is broken
    except Exception as e:
        print(f"Error checking scene {object_id}: {e}")

    return False


def generate_envir_map_dir(envmap_h, envmap_w):
    """Generate environment map directions and weights."""
    lat_step_size = np.pi / envmap_h
    lng_step_size = 2 * np.pi / envmap_w
    theta, phi = torch.meshgrid([
        torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, envmap_h),
        torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, envmap_w)
    ], indexing='ij')

    sin_theta = torch.sin(torch.pi / 2 - theta)
    light_area_weight = 4 * np.pi * sin_theta / torch.sum(sin_theta)
    light_area_weight = light_area_weight.to(torch.float32)

    view_dirs = torch.stack([
        torch.cos(phi) * torch.cos(theta),
        torch.sin(phi) * torch.cos(theta),
        torch.sin(theta)
    ], dim=-1).view(-1, 3)

    return light_area_weight, view_dirs


def get_light(envir_map, incident_dir, hdr_weight=None, if_weighted=False):
    """Sample light from environment map given incident direction."""
    try:
        envir_map = envir_map.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        if hdr_weight is not None:
            hdr_weight = hdr_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        incident_dir = incident_dir.clamp(-1, 1)
        theta = torch.arccos(incident_dir[:, 2]).reshape(-1)
        phi = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1)

        query_y = (theta / np.pi) * 2 - 1
        query_y = query_y.clamp(-1+1e-8, 1-1e-8)
        query_x = -phi / np.pi
        query_x = query_x.clamp(-1+1e-8, 1-1e-8)

        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0).float()

        if if_weighted and hdr_weight is not None:
            weighted_envir_map = envir_map * hdr_weight
            light_rgbs = F.grid_sample(weighted_envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
            light_rgbs = light_rgbs / hdr_weight.reshape(-1, 1)
        else:
            light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)

        return light_rgbs
    except Exception as e:
        print(f"Error in get_light: {e}")
        return None


def read_hdr_exr(path):
    """Read HDR EXR file."""
    if not HAS_PYEXR:
        raise ImportError("pyexr is required to read EXR files")
    try:
        rgb = pyexr.read(path)
        if rgb is not None and rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        return rgb
    except Exception as e:
        print(f"Error reading HDR file {path}: {e}")
        return None


def read_hdr(path):
    """Read HDR file (supports both .hdr and .exr)."""
    if path.endswith('.exr'):
        return read_hdr_exr(path)
    else:
        try:
            with open(path, 'rb') as h:
                buffer_ = np.frombuffer(h.read(), np.uint8)
            bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        except Exception as e:
            print(f"Error reading HDR file {path}: {e}")
            return None


def rotate_and_preprocess_envir_map(envir_map, camera_pose, euler_rotation=None, light_area_weight=None, view_dirs=None):
    """
    Rotate and preprocess environment map based on euler rotation and camera pose.
    Returns HDR raw, LDR, and HDR processed versions.
    """
    try:
        # Convert to numpy if it's a tensor
        if isinstance(envir_map, torch.Tensor):
            envir_map_np = envir_map.cpu().numpy()
        else:
            envir_map_np = envir_map.copy()

        env_h, env_w = envir_map_np.shape[0], envir_map_np.shape[1]
        if light_area_weight is None or view_dirs is None:
            light_area_weight, view_dirs = generate_envir_map_dir(env_h, env_w)

        # Store original for raw version
        envir_map_raw = envir_map_np.copy()

        # Step 1: Apply euler rotation (horizontal roll) if provided
        if euler_rotation is not None:
            z_rotation = euler_rotation[2] if len(euler_rotation) >= 3 else 0.0
            rotation_angle_deg = np.degrees(z_rotation)
            shift = int((rotation_angle_deg / 360.0) * env_w)
            envir_map_np = np.roll(envir_map_np, shift, axis=1)

        # Convert to tensor
        envir_map_tensor = torch.from_numpy(envir_map_np).float()

        # Step 2: Apply camera pose rotation
        if camera_pose.shape == (4, 4):
            c2w_rotation = camera_pose[:3, :3]
            w2c_rotation = c2w_rotation.T
        else:
            w2c_rotation = camera_pose.T

        # Blender's convention
        axis_aligned_transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        axis_aligned_R = axis_aligned_transform @ w2c_rotation
        view_dirs_world = view_dirs @ torch.from_numpy(axis_aligned_R).float()

        # Apply rotation using get_light
        rotated_hdr_rgb = get_light(envir_map_tensor, view_dirs_world.clamp(-1, 1))
        if rotated_hdr_rgb is not None and rotated_hdr_rgb.numel() > 0:
            rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3).cpu().numpy()
        else:
            rotated_hdr_rgb = envir_map_raw

        # HDR raw
        envir_map_hdr_raw = rotated_hdr_rgb

        # LDR (gamma correction)
        envir_map_ldr = rotated_hdr_rgb.clip(0, 1)
        envir_map_ldr = envir_map_ldr ** (1/2.2)

        # HDR processed (log transform)
        envir_map_hdr = np.log1p(10 * rotated_hdr_rgb)
        envir_map_hdr_rescaled = (envir_map_hdr / np.max(envir_map_hdr)).clip(0, 1)

        return envir_map_hdr_raw, envir_map_ldr, envir_map_hdr_rescaled
    except Exception as e:
        print(f"Error in rotate_and_preprocess_envir_map: {e}")
        return None, None, None


def create_tar_from_directory(source_dir, tar_path):
    """
    Create a tar archive from a directory.
    The tar file will contain all files from the directory, preserving the directory structure.

    Args:
        source_dir: Source directory to compress
        tar_path: Path to the output tar file
    """
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory {source_dir} does not exist, skipping tar creation")
        return

    # Check if there's a directory with the same name as the tar file (without .tar extension)
    # If it exists and is empty or only contains the same files, remove it
    tar_dir_name = tar_path.replace('.tar', '')
    if os.path.exists(tar_dir_name) and os.path.isdir(tar_dir_name):
        # Check if directory is empty or contains only the same files
        try:
            dir_contents = os.listdir(tar_dir_name)
            if len(dir_contents) == 0:
                # Empty directory, remove it
                print(f"Removing empty directory: {tar_dir_name}")
                try:
                    shutil.rmtree(tar_dir_name, ignore_errors=True)
                    if os.path.exists(tar_dir_name):
                        # Try using rm -rf if shutil fails
                        subprocess.run(['rm', '-rf', tar_dir_name], check=False)
                except Exception as e:
                    print(f"Warning: Could not remove directory {tar_dir_name}: {e}")
        except Exception as e:
            print(f"Warning: Could not check directory contents {tar_dir_name}: {e}")

    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    with tarfile.open(tar_path, 'w') as tar:
        # Add all files in the directory, preserving relative paths
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Get relative path from source_dir
                arcname = os.path.relpath(file_path, source_dir)
                tar.add(file_path, arcname=arcname)
    print(f"Created tar archive: {tar_path}")


def _resolve_light_source(split_path, folder_name):
    """Return ('dir'|'tar'|None, source_path)."""
    dir_path = os.path.join(split_path, folder_name)
    tar_path = os.path.join(split_path, f"{folder_name}.tar")
    if os.path.isdir(dir_path):
        return "dir", dir_path
    if os.path.isfile(tar_path):
        return "tar", tar_path
    return None, None


def _list_numeric_gt_images(split_path, folder_name):
    """
    List (frame_idx, filename) for gt_*.png in a folder or tar.
    """
    source_type, source_path = _resolve_light_source(split_path, folder_name)
    if source_type is None:
        return []

    all_image_files = []
    if source_type == "dir":
        all_image_files = [f for f in os.listdir(source_path) if f.startswith("gt_") and f.endswith(".png")]
    else:
        with tarfile.open(source_path, "r") as tar:
            all_image_files = [
                os.path.basename(m.name)
                for m in tar.getmembers()
                if m.isfile() and os.path.basename(m.name).startswith("gt_") and os.path.basename(m.name).endswith(".png")
            ]

    image_files_with_idx = []
    for image_file in all_image_files:
        idx_str = image_file.replace("gt_", "").replace(".png", "")
        try:
            frame_idx = int(idx_str)
            image_files_with_idx.append((frame_idx, image_file))
        except ValueError:
            continue
    image_files_with_idx.sort(key=lambda x: x[0])
    return image_files_with_idx


def _read_image_size_from_source(split_path, folder_name, image_name):
    source_type, source_path = _resolve_light_source(split_path, folder_name)
    if source_type is None:
        raise FileNotFoundError(f"Missing source for {folder_name} under {split_path}")

    if source_type == "dir":
        image_path = os.path.join(source_path, image_name)
        with Image.open(image_path) as img:
            return img.size

    with tarfile.open(source_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and os.path.basename(m.name) == image_name]
        if not members:
            raise FileNotFoundError(f"{image_name} not found in tar {source_path}")
        member = members[0]
        fobj = tar.extractfile(member)
        if fobj is None:
            raise FileNotFoundError(f"Could not extract {image_name} from {source_path}")
        with Image.open(io.BytesIO(fobj.read())) as img:
            return img.size


def _copy_image_from_source(split_path, folder_name, image_name, output_image_path):
    source_type, source_path = _resolve_light_source(split_path, folder_name)
    if source_type is None:
        raise FileNotFoundError(f"Missing source for {folder_name} under {split_path}")

    if source_type == "dir":
        shutil.copy2(os.path.join(source_path, image_name), output_image_path)
        return

    with tarfile.open(source_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and os.path.basename(m.name) == image_name]
        if not members:
            raise FileNotFoundError(f"{image_name} not found in tar {source_path}")
        member = members[0]
        fobj = tar.extractfile(member)
        if fobj is None:
            raise FileNotFoundError(f"Could not extract {image_name} from {source_path}")
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        with open(output_image_path, "wb") as out_f:
            out_f.write(fobj.read())


def _load_json_from_source(split_path, folder_name, json_names):
    """
    Load first existing json from source folder/tar.
    Returns dict or None.
    """
    source_type, source_path = _resolve_light_source(split_path, folder_name)
    if source_type is None:
        return None

    if source_type == "dir":
        for name in json_names:
            path = os.path.join(source_path, name)
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
        return None

    with tarfile.open(source_path, "r") as tar:
        file_members = [m for m in tar.getmembers() if m.isfile()]
        member_by_name = {m.name: m for m in file_members}
        member_by_base = {os.path.basename(m.name): m for m in file_members}
        for name in json_names:
            member = member_by_name.get(name) or member_by_base.get(name)
            if member is not None:
                fobj = tar.extractfile(member)
                if fobj is not None:
                    return json.load(fobj)
        return None


def check_scene_exists_in_outputs(scene_name, output_root, output_tar_root, split, is_point_light_scene=False):
    """
    Check if a scene already exists in both output locations (A and B).

    Args:
        scene_name: Scene name to check
        output_root: Location A (uncompressed)
        output_tar_root: Location B (compressed, can be None)
        split: 'train' or 'test'
        is_point_light_scene: If True, check point_light_rays instead of envmaps.

    Returns:
        bool: True if scene exists in both locations (or only location A if output_tar_root is None), False otherwise
    """
    # Check location A: metadata JSON file should exist
    metadata_json = os.path.join(output_root, split, 'metadata', f"{scene_name}.json")
    exists_in_a = os.path.exists(metadata_json)
    if is_point_light_scene:
        pl_rays_path = os.path.join(output_root, split, 'point_light_rays', f"{scene_name}.npy")
        exists_in_a = exists_in_a and os.path.exists(pl_rays_path)
    # If output_tar_root is None, only check location A
    if output_tar_root is None:
        return exists_in_a
    # Check location B: tar files should exist in images, (envmaps or point_light_rays), and albedos folders
    images_tar = os.path.join(output_tar_root, split, 'images', f"{scene_name}.tar")
    object_id = scene_name.split('_')[0]
    albedos_tar = os.path.join(output_tar_root, split, 'albedos', f"{object_id}.tar")
    if is_point_light_scene:
        pl_tar = os.path.join(output_tar_root, split, 'point_light_rays', f"{scene_name}.tar")
        exists_in_b = (os.path.exists(images_tar) and os.path.exists(pl_tar) and os.path.exists(albedos_tar))
    else:
        envmaps_tar = os.path.join(output_tar_root, split, 'envmaps', f"{scene_name}.tar")
        exists_in_b = (os.path.exists(images_tar) and
                       os.path.exists(envmaps_tar) and
                       os.path.exists(albedos_tar))
    return exists_in_a and exists_in_b


def process_objaverse_scene(objaverse_root, object_id, output_root, output_tar_root, split='test', hdri_dir=None, point_light_rays_n=8192, scene_sphere_radius=3.0):
    """
    Process a single Objaverse object and convert all lighting scenes to re10k format.

    Supported folder types:
      - env_*/white_env_*: environment-lit scenes (envmaps)
      - rgb_pl_*/white_pl_*: single point-light scenes
      - multi_pl_*: multiple point-light scenes
      - area_*: area light scenes
      - combined_*: combined lighting (env + point/area combos)

    Args:
        objaverse_root: Root directory of objaverse data
        object_id: Object ID folder name
        output_root: Root directory for output
        output_tar_root: Root directory for tar archives (location B, can be None)
        split: 'train' or 'test'
        hdri_dir: Directory containing HDR environment maps (optional)
        point_light_rays_n: Number of rays to sample per scene (default 8192)
        scene_sphere_radius: Radius of the scene bounding sphere for ray sampling (default 3.0)
    """
    object_path = os.path.join(objaverse_root, object_id)
    split_path = os.path.join(object_path, split)

    if not os.path.exists(split_path):
        print(f"Warning: {split_path} does not exist, skipping")
        return None

    # Check if scene is broken (no materials)
    if check_scene_broken(split_path, object_id):
        print(f"Scene {object_id} is broken (no materials), marking...")
        broken_file = os.path.join(object_path, 'broken.txt')
        with open(broken_file, 'w') as f:
            f.write("broken\n")
        return "broken"

    # Load cameras.json
    cameras_json_path = os.path.join(split_path, 'cameras.json')
    if not os.path.exists(cameras_json_path):
        print(f"Warning: {cameras_json_path} does not exist, skipping {object_id}")
        return None

    with open(cameras_json_path, 'r') as f:
        cameras_data = json.load(f)

    # Find all lighting folders and tar files
    env_folders = []
    env_tar_files = []
    pl_folders = []
    pl_tar_files = []
    multi_pl_folders = []
    area_folders = []
    combined_folders = []
    for item in os.listdir(split_path):
        item_path = os.path.join(split_path, item)
        if os.path.isdir(item_path):
            if item.startswith('env_') or item.startswith('white_env_'):
                env_folders.append(item)
            elif is_pl_folder(item):
                pl_folders.append(item)
            elif is_multi_pl_folder(item):
                multi_pl_folders.append(item)
            elif is_area_folder(item):
                area_folders.append(item)
            elif is_combined_folder(item):
                combined_folders.append(item)
        elif item.endswith('.tar'):
            folder_name = item.replace('.tar', '')
            if folder_name.startswith('env_') or folder_name.startswith('white_env_'):
                env_tar_files.append(folder_name)
            elif is_pl_folder(folder_name):
                pl_tar_files.append(folder_name)

    all_light_folders = (
        sorted(set(env_folders + env_tar_files))
        + sorted(set(pl_folders + pl_tar_files))
        + sorted(multi_pl_folders)
        + sorted(area_folders)
        + sorted(combined_folders)
    )
    if not all_light_folders:
        # All folders are already compressed or none found; collect existing scenes from tar
        scene_names_from_tar = []
        for env_folder_name in env_tar_files:
            scene_name = f"{object_id}_{env_folder_name}"
            if check_scene_exists_in_outputs(scene_name, output_root, output_tar_root, split, is_point_light_scene=False):
                scene_names_from_tar.append(scene_name)
        for pl_folder_name in pl_tar_files:
            scene_name = f"{object_id}_{pl_folder_name}"
            if check_scene_exists_in_outputs(scene_name, output_root, output_tar_root, split, is_point_light_scene=True):
                scene_names_from_tar.append(scene_name)
        if scene_names_from_tar:
            return scene_names_from_tar
        print(f"Warning: No lighting folders found in {split_path}, skipping {object_id}")
        return None

    # Process albedo folder (shared across all scenes with the same object_id)
    # First, get the number of frames from the first light folder (env or pl) to check if albedo is complete
    first_light_folder = all_light_folders[0]
    first_env_image_files_with_idx = _list_numeric_gt_images(split_path, first_light_folder)
    num_frames = len(first_env_image_files_with_idx) if first_env_image_files_with_idx else 0

    # Check and process albedo folder
    output_albedos_dir = os.path.join(output_root, split, 'albedos', object_id)
    source_albedo_dir = os.path.join(split_path, 'albedo')
    source_albedo_tar = os.path.join(split_path, 'albedo.tar')

    # Check if source is a tar file (already compressed)
    source_is_tar = os.path.exists(source_albedo_tar) and not os.path.exists(source_albedo_dir)

    should_process_albedo = False
    if not os.path.exists(output_albedos_dir):
        if source_is_tar:
            # If source is tar and output doesn't exist, check if tar exists in output_tar_root
            if output_tar_root:
                albedos_tar_path = os.path.join(output_tar_root, split, 'albedos', f"{object_id}.tar")
                if os.path.exists(albedos_tar_path):
                    print(f"Albedo tar for {object_id} already exists in location B, skipping")
                else:
                    print(f"Warning: Source albedo is tar but output doesn't exist, need to extract first")
            else:
                print(f"Warning: Source albedo is tar but no output_tar_root specified")
        else:
            should_process_albedo = True
            print(f"Albedo folder for {object_id} does not exist, will process")
    elif source_is_tar:
        print(f"Source albedo is already compressed as tar, output exists, skipping albedo processing")
    elif not os.path.exists(source_albedo_dir):
        print(f"Warning: Source albedo directory {source_albedo_dir} does not exist, skipping albedo processing")
    else:
        # Check if albedo folder is fully populated
        # Get the expected number of albedo files from source directory
        if os.path.exists(source_albedo_dir):
            source_albedo_files = [f for f in os.listdir(source_albedo_dir)
                                 if f.endswith('.png') and f.startswith('albedo_cam_')]
            expected_num_albedos = len(source_albedo_files)
        else:
            expected_num_albedos = num_frames

        # Check existing files in output directory (should be in 00000.png format)
        existing_albedo_files = [f for f in os.listdir(output_albedos_dir)
                                 if f.endswith('.png') and len(f) == 9 and f[:5].isdigit()]
        if len(existing_albedo_files) < expected_num_albedos:
            should_process_albedo = True
            print(f"Albedo folder for {object_id} is not fully populated ({len(existing_albedo_files)}/{expected_num_albedos} files), will process")
        else:
            print(f"Albedo folder for {object_id} already exists and is fully populated, skipping")

    # Process albedo if needed
    if should_process_albedo and os.path.exists(source_albedo_dir):
        os.makedirs(output_albedos_dir, exist_ok=True)
        source_albedo_files = sorted([f for f in os.listdir(source_albedo_dir)
                                     if f.endswith('.png')])

        # Copy all albedo files with renamed format (albedo_cam_0.png -> 00000.png)
        copied_count = 0
        for albedo_file in source_albedo_files:
            # Extract camera index from filename (e.g., albedo_cam_0.png -> 0)
            if albedo_file.startswith('albedo_cam_'):
                # Extract the index part
                idx_str = albedo_file.replace('albedo_cam_', '').replace('.png', '')
                try:
                    cam_idx = int(idx_str)
                    # Convert to zero-padded format (5 digits) to match images format
                    output_albedo_name = f"{cam_idx:05d}.png"
                    source_albedo_path = os.path.join(source_albedo_dir, albedo_file)
                    output_albedo_path = os.path.join(output_albedos_dir, output_albedo_name)
                    try:
                        shutil.copy2(source_albedo_path, output_albedo_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to copy albedo file {albedo_file}: {e}")
                except ValueError:
                    print(f"Warning: Could not extract camera index from {albedo_file}, skipping")
            else:
                print(f"Warning: Unexpected albedo filename format: {albedo_file}, skipping")

        print(f"Processed albedo folder for {object_id}: copied {copied_count} files")

    # Process each env folder as a separate scene
    processed_scene_names = []  # Track all processed scene names (including skipped ones)
    for env_folder in sorted(set(env_folders + env_tar_files)):
        source_type, source_path = _resolve_light_source(split_path, env_folder)
        if source_type is None:
            continue
        env_path = source_path

        # Find all gt_*.png images and filter to only numeric indices
        image_files_with_idx = _list_numeric_gt_images(split_path, env_folder)

        if not image_files_with_idx:
            print(f"Warning: No valid gt_*.png images with numeric indices found in {env_path}, skipping")
            continue

        # Sort by frame index
        image_files_with_idx.sort(key=lambda x: x[0])

        # Get image dimensions from first image
        try:
            image_width, image_height = _read_image_size_from_source(split_path, env_folder, image_files_with_idx[0][1])
        except Exception as e:
            print(f"Error reading first image for {env_folder}: {e}, skipping")
            continue

        # Create scene name: object_id_env_folder
        scene_name = f"{object_id}_{env_folder}"

        # Create output directories
        output_metadata_dir = os.path.join(output_root, split, 'metadata')
        output_images_dir = os.path.join(output_root, split, 'images', scene_name)
        output_envmaps_dir = os.path.join(output_root, split, 'envmaps', scene_name)
        os.makedirs(output_metadata_dir, exist_ok=True)

        # Load environment map info if available (needed to check if envmaps should exist)
        # For train split, prefer matching source under test split.
        env_info = None
        if split == 'train':
            test_split_path = os.path.join(object_path, 'test')
            if os.path.exists(test_split_path):
                env_info = _load_json_from_source(test_split_path, env_folder, ['env.json', 'white_env.json'])
        if env_info is None:
            env_info = _load_json_from_source(split_path, env_folder, ['env.json', 'white_env.json'])

        # Check if scene already exists and all files are present
        output_json_path = os.path.join(output_metadata_dir, f"{scene_name}.json")
        # Check if output directories exist (we don't check json existence to allow regeneration)
        scene_exists = os.path.exists(output_images_dir) and os.path.exists(output_envmaps_dir)

        # Initialize skip_file_processing flag
        skip_file_processing = False

        if scene_exists:
            # Check if all expected files exist
            all_files_exist = True

            # Check if all image files exist
            for frame_idx, image_file in image_files_with_idx:
                output_image_name = f"{frame_idx:05d}.png"
                output_image_path = os.path.join(output_images_dir, output_image_name)
                if not os.path.exists(output_image_path):
                    all_files_exist = False
                    break

            # Check if environment maps exist (if env_info is available and hdri_dir is provided)
            if all_files_exist and env_info and hdri_dir:
                for frame_idx, image_file in image_files_with_idx:
                    output_envmap_hdr_name = f"{frame_idx:05d}_hdr.png"
                    output_envmap_ldr_name = f"{frame_idx:05d}_ldr.png"
                    output_envmap_hdr_path = os.path.join(output_envmaps_dir, output_envmap_hdr_name)
                    output_envmap_ldr_path = os.path.join(output_envmaps_dir, output_envmap_ldr_name)
                    if not (os.path.exists(output_envmap_hdr_path) and os.path.exists(output_envmap_ldr_path)):
                        all_files_exist = False
                        break

            if all_files_exist:
                print(f"Skipping image/envmap processing for {scene_name}: all files already exist")
                # Still generate and save JSON file to ensure it's up-to-date and not corrupted
                # This will regenerate the JSON even if files exist
                skip_file_processing = True
            else:
                skip_file_processing = False

        # Create directories if they don't exist
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_envmaps_dir, exist_ok=True)

        # Load environment map if available (only if not skipping file processing)
        env_map = None
        euler_rotation = None
        if not skip_file_processing and env_info and hdri_dir:
            env_map_name = env_info.get('env_map', '')
            if env_map_name:
                # Try different filename variations (with/without extension, with _8k suffix, etc.)
                possible_names = [
                    env_map_name,  # Original name as-is
                    f"{env_map_name}.exr",  # Add .exr extension
                    f"{env_map_name}.hdr",  # Add .hdr extension
                    f"{env_map_name}_8k.exr",  # Add _8k.exr suffix
                    f"{env_map_name}_8k.hdr",  # Add _8k.hdr suffix
                ]

                env_map_path = None
                for name in possible_names:
                    test_path = os.path.join(hdri_dir, name)
                    if os.path.exists(test_path):
                        env_map_path = test_path
                        break

                if env_map_path:
                    env_map = read_hdr(env_map_path)
                    if env_map is not None:
                        env_map = torch.from_numpy(env_map).float()
                        euler_rotation = env_info.get('rotation_euler', None)
                    else:
                        print(f"Warning: Failed to read environment map {env_map_path}")
                else:
                    print(f"Warning: Environment map '{env_map_name}' not found in {hdri_dir} (tried: {', '.join(possible_names)})")

        # Generate environment map directions if needed
        light_area_weight = None
        view_dirs = None
        if env_map is not None:
            env_h, env_w = env_map.shape[0], env_map.shape[1]
            light_area_weight, view_dirs = generate_envir_map_dir(env_h, env_w)

        # Process frames
        frames = []
        for frame_idx, image_file in image_files_with_idx:
            # Find corresponding camera in cameras.json using frame_idx
            if frame_idx < len(cameras_data):
                camera_info = cameras_data[frame_idx]
            else:
                print(f"Warning: Frame {frame_idx} not found in cameras.json (only {len(cameras_data)} cameras available), skipping")
                continue

            # Convert c2w from Blender to OpenCV
            c2w_blender = np.array(camera_info['c2w'])
            c2w_opencv = blender_to_opencv_c2w(c2w_blender)

            # Convert to w2c (world-to-camera)
            w2c = np.linalg.inv(c2w_opencv)

            # Convert FOV to fxfycxcy
            fov = camera_info.get('fov', 30.0)  # Default to 30 if not specified
            fxfycxcy = fov_to_fxfycxcy(fov, image_width, image_height)

            # Determine output image path (used for JSON even if we skip copying)
            output_image_name = f"{frame_idx:05d}.png"
            output_image_path = os.path.join(output_images_dir, output_image_name)

            # Only copy image and process envmaps if not skipping
            if not skip_file_processing:
                # Copy image to output directory with zero-padded name
                _copy_image_from_source(split_path, env_folder, image_file, output_image_path)

                # Process environment map if available
                if env_map is not None:
                    # Get camera pose (c2w) - use Blender format before OpenCV conversion
                    c2w_blender = np.array(camera_info['c2w'])

                    # Rotate and preprocess environment map (using Blender format c2w)
                    env_hdr_raw, env_ldr, env_hdr = rotate_and_preprocess_envir_map(
                        env_map, c2w_blender, euler_rotation=euler_rotation,
                        light_area_weight=light_area_weight, view_dirs=view_dirs
                    )

                    if env_hdr_raw is not None:
                        # Save HDR and LDR versions separately
                        # HDR version: log transform and rescale
                        env_hdr_uint8 = np.uint8(env_hdr * 255)
                        env_hdr_img = Image.fromarray(env_hdr_uint8)
                        output_envmap_hdr_name = f"{frame_idx:05d}_hdr.png"
                        output_envmap_hdr_path = os.path.join(output_envmaps_dir, output_envmap_hdr_name)
                        env_hdr_img.save(output_envmap_hdr_path)

                        # LDR version: gamma correction
                        env_ldr_uint8 = np.uint8(env_ldr * 255)
                        env_ldr_img = Image.fromarray(env_ldr_uint8)
                        output_envmap_ldr_name = f"{frame_idx:05d}_ldr.png"
                        output_envmap_ldr_path = os.path.join(output_envmaps_dir, output_envmap_ldr_name)
                        env_ldr_img.save(output_envmap_ldr_path)

            # Create absolute image path for the JSON file
            # This ensures the path works regardless of where the code is run from
            absolute_image_path = os.path.abspath(output_image_path)

            # Create frame entry
            frame = {
                "image_path": absolute_image_path,
                "fxfycxcy": fxfycxcy,
                "w2c": w2c.tolist()
            }
            frames.append(frame)

        # Create scene JSON
        scene_data = {
            "scene_name": scene_name,
            "frames": frames
        }

        # Save scene JSON (always regenerate, even if files were skipped)
        output_json_path = os.path.join(output_metadata_dir, f"{scene_name}.json")
        with open(output_json_path, 'w') as f:
            json.dump(scene_data, f, indent=2)

        if skip_file_processing:
            print(f"Regenerated JSON for {scene_name}: {len(frames)} frames (skipped file processing)")
        else:
            print(f"Processed {scene_name}: {len(frames)} frames")
        processed_scene_names.append(scene_name)  # Add to processed list

        # Create tar archives for location B after processing each scene
        if output_tar_root is not None and not skip_file_processing:
            # Create tar for images folder
            images_tar_path = os.path.join(output_tar_root, split, 'images', f"{scene_name}.tar")
            images_tar_dir = images_tar_path.replace('.tar', '')
            # Check if there's an empty directory with the same name (without .tar)
            if os.path.exists(images_tar_dir) and os.path.isdir(images_tar_dir):
                try:
                    if len(os.listdir(images_tar_dir)) == 0:
                        # Empty directory, remove it
                        print(f"Removing empty directory: {images_tar_dir}")
                        shutil.rmtree(images_tar_dir, ignore_errors=True)
                        if os.path.exists(images_tar_dir):
                            subprocess.run(['rm', '-rf', images_tar_dir], check=False)
                except Exception:
                    pass
            if not os.path.exists(images_tar_path):
                create_tar_from_directory(output_images_dir, images_tar_path)

            # Create tar for envmaps folder
            envmaps_tar_path = os.path.join(output_tar_root, split, 'envmaps', f"{scene_name}.tar")
            envmaps_tar_dir = envmaps_tar_path.replace('.tar', '')
            # Check if there's an empty directory with the same name (without .tar)
            if os.path.exists(envmaps_tar_dir) and os.path.isdir(envmaps_tar_dir):
                try:
                    if len(os.listdir(envmaps_tar_dir)) == 0:
                        # Empty directory, remove it
                        print(f"Removing empty directory: {envmaps_tar_dir}")
                        shutil.rmtree(envmaps_tar_dir, ignore_errors=True)
                        if os.path.exists(envmaps_tar_dir):
                            subprocess.run(['rm', '-rf', envmaps_tar_dir], check=False)
                except Exception:
                    pass
            if not os.path.exists(envmaps_tar_path):
                create_tar_from_directory(output_envmaps_dir, envmaps_tar_path)

            # Create metadata JSON for location B with updated image paths
            output_metadata_dir_b = os.path.join(output_tar_root, split, 'metadata')
            os.makedirs(output_metadata_dir_b, exist_ok=True)
            output_json_path_b = os.path.join(output_metadata_dir_b, f"{scene_name}.json")

            # Create a copy of scene_data with updated image paths pointing to location B
            scene_data_b = {
                "scene_name": scene_name,
                "frames": []
            }

            # Get absolute path for images tar file
            images_tar_abs_path = os.path.abspath(images_tar_path)

            # Update image paths in frames to point to location B (tar file path + internal path)
            for frame in frames:
                frame_b = frame.copy()
                # Extract the frame filename from the original image_path
                original_image_path = frame['image_path']
                frame_filename = os.path.basename(original_image_path)
                # Create path pointing to tar file with internal path
                # The tar file contains files directly (not in a subdirectory), so internal path is just the filename
                # Format: {tar_absolute_path}::{internal_path}
                frame_b['image_path'] = f"{images_tar_abs_path}::{frame_filename}"
                scene_data_b['frames'].append(frame_b)

            # Save metadata JSON for location B
            with open(output_json_path_b, 'w') as f:
                json.dump(scene_data_b, f, indent=2)
            print(f"Created metadata JSON for location B: {output_json_path_b}")

        # The public preprocessor deliberately preserves the raw renderer output.

    # Process each point-light folder as a separate scene (images + metadata + point_light_rays .npy)
    for pl_folder in sorted(set(pl_folders + pl_tar_files)):
        source_type, source_path = _resolve_light_source(split_path, pl_folder)
        if source_type is None:
            continue
        pl_path = source_path
        image_files_with_idx = _list_numeric_gt_images(split_path, pl_folder)
        if not image_files_with_idx:
            print(f"Warning: No valid gt_*.png in {pl_path}, skipping")
            continue
        image_files_with_idx.sort(key=lambda x: x[0])
        try:
            image_width, image_height = _read_image_size_from_source(split_path, pl_folder, image_files_with_idx[0][1])
        except Exception as e:
            print(f"Error reading first image for {pl_folder}: {e}, skipping")
            continue
        scene_name = f"{object_id}_{pl_folder}"
        output_metadata_dir = os.path.join(output_root, split, 'metadata')
        output_images_dir = os.path.join(output_root, split, 'images', scene_name)
        output_point_light_rays_dir = os.path.join(output_root, split, 'point_light_rays')
        output_pl_rays_path = os.path.join(output_point_light_rays_dir, f"{scene_name}.npy")
        os.makedirs(output_metadata_dir, exist_ok=True)
        scene_exists = os.path.exists(output_images_dir) and os.path.exists(output_pl_rays_path)
        skip_file_processing = False
        if scene_exists:
            all_files_exist = True
            for frame_idx, image_file in image_files_with_idx:
                output_image_path = os.path.join(output_images_dir, f"{frame_idx:05d}.png")
                if not os.path.exists(output_image_path):
                    all_files_exist = False
                    break
            if all_files_exist:
                print(f"Skipping point-light processing for {scene_name}: all files already exist")
                skip_file_processing = True
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_point_light_rays_dir, exist_ok=True)
        pl_info = load_point_light_info(pl_path)
        if pl_info is None and not skip_file_processing:
            print(f"Warning: No rgb_pl.json or white_pl.json in {pl_path}, skipping point-light rays")
        elif pl_info is not None:
            pos, color, power = pl_info
            if not skip_file_processing:
                rays_arr = build_point_light_rays_array(pos, color, power, N=point_light_rays_n, scene_sphere_radius=scene_sphere_radius)
                np.save(output_pl_rays_path, rays_arr)
        frames = []
        for frame_idx, image_file in image_files_with_idx:
            if frame_idx >= len(cameras_data):
                print(f"Warning: Frame {frame_idx} not in cameras.json, skipping")
                continue
            camera_info = cameras_data[frame_idx]
            c2w_blender = np.array(camera_info['c2w'])
            c2w_opencv = blender_to_opencv_c2w(c2w_blender)
            w2c = np.linalg.inv(c2w_opencv)
            fov = camera_info.get('fov', 30.0)
            fxfycxcy = fov_to_fxfycxcy(fov, image_width, image_height)
            output_image_name = f"{frame_idx:05d}.png"
            output_image_path = os.path.join(output_images_dir, output_image_name)
            if not skip_file_processing:
                _copy_image_from_source(split_path, pl_folder, image_file, output_image_path)
            absolute_image_path = os.path.abspath(output_image_path)
            frames.append({
                "image_path": absolute_image_path,
                "fxfycxcy": fxfycxcy,
                "w2c": w2c.tolist()
            })
        scene_data = {"scene_name": scene_name, "frames": frames}
        output_json_path = os.path.join(output_metadata_dir, f"{scene_name}.json")
        with open(output_json_path, 'w') as f:
            json.dump(scene_data, f, indent=2)
        print(f"Processed point-light scene {scene_name}: {len(frames)} frames")
        processed_scene_names.append(scene_name)
        if output_tar_root is not None and not skip_file_processing:
            images_tar_path = os.path.join(output_tar_root, split, 'images', f"{scene_name}.tar")
            if not os.path.exists(images_tar_path):
                create_tar_from_directory(output_images_dir, images_tar_path)
            pl_tar_path = os.path.join(output_tar_root, split, 'point_light_rays', f"{scene_name}.tar")
            os.makedirs(os.path.dirname(pl_tar_path), exist_ok=True)
            if not os.path.exists(pl_tar_path):
                with tarfile.open(pl_tar_path, 'w') as tar:
                    tar.add(output_pl_rays_path, arcname=os.path.basename(output_pl_rays_path))
                print(f"Created point_light_rays tar: {pl_tar_path}")
            output_metadata_dir_b = os.path.join(output_tar_root, split, 'metadata')
            os.makedirs(output_metadata_dir_b, exist_ok=True)
            output_json_path_b = os.path.join(output_metadata_dir_b, f"{scene_name}.json")
            scene_data_b = {"scene_name": scene_name, "frames": []}
            images_tar_abs_path = os.path.abspath(images_tar_path)
            for frame in frames:
                frame_b = frame.copy()
                frame_b['image_path'] = f"{images_tar_abs_path}::{os.path.basename(frame['image_path'])}"
                scene_data_b['frames'].append(frame_b)
            with open(output_json_path_b, 'w') as f:
                json.dump(scene_data_b, f, indent=2)
        # The public preprocessor deliberately preserves the raw renderer output.

    # ======================================================================
    # Process multi_pl_* folders (multiple point lights, no envmap)
    # ======================================================================
    for mpl_folder in sorted(multi_pl_folders):
        mpl_path = os.path.join(split_path, mpl_folder)
        all_image_files = [f for f in os.listdir(mpl_path) if f.startswith('gt_') and f.endswith('.png')]
        image_files_with_idx = []
        for image_file in all_image_files:
            idx_str = image_file.replace('gt_', '').replace('.png', '')
            try:
                frame_idx = int(idx_str)
                image_files_with_idx.append((frame_idx, image_file))
            except ValueError:
                continue
        if not image_files_with_idx:
            print(f"Warning: No valid gt_*.png in {mpl_path}, skipping")
            continue
        image_files_with_idx.sort(key=lambda x: x[0])
        first_image_path = os.path.join(mpl_path, image_files_with_idx[0][1])
        try:
            with Image.open(first_image_path) as img:
                image_width, image_height = img.size
        except Exception as e:
            print(f"Error reading image {first_image_path}: {e}, skipping")
            continue

        scene_name = f"{object_id}_{mpl_folder}"
        output_metadata_dir = os.path.join(output_root, split, 'metadata')
        output_images_dir = os.path.join(output_root, split, 'images', scene_name)
        output_point_light_rays_dir = os.path.join(output_root, split, 'point_light_rays')
        output_pl_rays_path = os.path.join(output_point_light_rays_dir, f"{scene_name}.npy")
        os.makedirs(output_metadata_dir, exist_ok=True)

        scene_exists = os.path.exists(output_images_dir) and os.path.exists(output_pl_rays_path)
        skip_file_processing = False
        if scene_exists:
            all_files_exist = all(
                os.path.exists(os.path.join(output_images_dir, f"{fi:05d}.png"))
                for fi, _ in image_files_with_idx
            )
            if all_files_exist:
                print(f"Skipping multi-pl processing for {scene_name}: all files already exist")
                skip_file_processing = True

        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_point_light_rays_dir, exist_ok=True)

        mpl_info = load_multi_point_light_info(mpl_path)
        if mpl_info is None and not skip_file_processing:
            print(f"Warning: No multi_pl.json in {mpl_path}, skipping point-light rays")
        elif mpl_info is not None and not skip_file_processing:
            rays_arr = build_multi_source_light_rays(
                point_lights=mpl_info, area_lights=[],
                N=point_light_rays_n, scene_sphere_radius=scene_sphere_radius
            )
            np.save(output_pl_rays_path, rays_arr)

        frames = []
        for frame_idx, image_file in image_files_with_idx:
            if frame_idx >= len(cameras_data):
                continue
            camera_info = cameras_data[frame_idx]
            c2w_blender = np.array(camera_info['c2w'])
            c2w_opencv = blender_to_opencv_c2w(c2w_blender)
            w2c = np.linalg.inv(c2w_opencv)
            fov = camera_info.get('fov', 30.0)
            fxfycxcy = fov_to_fxfycxcy(fov, image_width, image_height)
            output_image_name = f"{frame_idx:05d}.png"
            output_image_path = os.path.join(output_images_dir, output_image_name)
            if not skip_file_processing:
                shutil.copy2(os.path.join(mpl_path, image_file), output_image_path)
            frames.append({
                "image_path": os.path.abspath(output_image_path),
                "fxfycxcy": fxfycxcy,
                "w2c": w2c.tolist()
            })

        scene_data = {"scene_name": scene_name, "frames": frames}
        with open(os.path.join(output_metadata_dir, f"{scene_name}.json"), 'w') as f:
            json.dump(scene_data, f, indent=2)
        print(f"Processed multi-pl scene {scene_name}: {len(frames)} frames")
        processed_scene_names.append(scene_name)

    # ======================================================================
    # Process area_* folders (area light, no envmap)
    # ======================================================================
    for area_folder in sorted(area_folders):
        area_path = os.path.join(split_path, area_folder)
        all_image_files = [f for f in os.listdir(area_path) if f.startswith('gt_') and f.endswith('.png')]
        image_files_with_idx = []
        for image_file in all_image_files:
            idx_str = image_file.replace('gt_', '').replace('.png', '')
            try:
                frame_idx = int(idx_str)
                image_files_with_idx.append((frame_idx, image_file))
            except ValueError:
                continue
        if not image_files_with_idx:
            print(f"Warning: No valid gt_*.png in {area_path}, skipping")
            continue
        image_files_with_idx.sort(key=lambda x: x[0])
        first_image_path = os.path.join(area_path, image_files_with_idx[0][1])
        try:
            with Image.open(first_image_path) as img:
                image_width, image_height = img.size
        except Exception as e:
            print(f"Error reading image {first_image_path}: {e}, skipping")
            continue

        scene_name = f"{object_id}_{area_folder}"
        output_metadata_dir = os.path.join(output_root, split, 'metadata')
        output_images_dir = os.path.join(output_root, split, 'images', scene_name)
        output_point_light_rays_dir = os.path.join(output_root, split, 'point_light_rays')
        output_pl_rays_path = os.path.join(output_point_light_rays_dir, f"{scene_name}.npy")
        os.makedirs(output_metadata_dir, exist_ok=True)

        scene_exists = os.path.exists(output_images_dir) and os.path.exists(output_pl_rays_path)
        skip_file_processing = False
        if scene_exists:
            all_files_exist = all(
                os.path.exists(os.path.join(output_images_dir, f"{fi:05d}.png"))
                for fi, _ in image_files_with_idx
            )
            if all_files_exist:
                print(f"Skipping area-light processing for {scene_name}: all files already exist")
                skip_file_processing = True

        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_point_light_rays_dir, exist_ok=True)

        al_info = load_area_light_info(area_path)
        if al_info is None and not skip_file_processing:
            print(f"Warning: No area.json in {area_path}, skipping area-light rays")
        elif al_info is not None and not skip_file_processing:
            pos, color, power, size = al_info
            rays_arr = build_area_light_rays_array(
                pos, color, power, size,
                N=point_light_rays_n, scene_sphere_radius=scene_sphere_radius
            )
            np.save(output_pl_rays_path, rays_arr)

        frames = []
        for frame_idx, image_file in image_files_with_idx:
            if frame_idx >= len(cameras_data):
                continue
            camera_info = cameras_data[frame_idx]
            c2w_blender = np.array(camera_info['c2w'])
            c2w_opencv = blender_to_opencv_c2w(c2w_blender)
            w2c = np.linalg.inv(c2w_opencv)
            fov = camera_info.get('fov', 30.0)
            fxfycxcy = fov_to_fxfycxcy(fov, image_width, image_height)
            output_image_name = f"{frame_idx:05d}.png"
            output_image_path = os.path.join(output_images_dir, output_image_name)
            if not skip_file_processing:
                shutil.copy2(os.path.join(area_path, image_file), output_image_path)
            frames.append({
                "image_path": os.path.abspath(output_image_path),
                "fxfycxcy": fxfycxcy,
                "w2c": w2c.tolist()
            })

        scene_data = {"scene_name": scene_name, "frames": frames}
        with open(os.path.join(output_metadata_dir, f"{scene_name}.json"), 'w') as f:
            json.dump(scene_data, f, indent=2)
        print(f"Processed area-light scene {scene_name}: {len(frames)} frames")
        processed_scene_names.append(scene_name)

    # ======================================================================
    # Process combined_* folders (env + point/area combos)
    # ======================================================================
    for comb_folder in sorted(combined_folders):
        comb_path = os.path.join(split_path, comb_folder)
        all_image_files = [f for f in os.listdir(comb_path) if f.startswith('gt_') and f.endswith('.png')]
        image_files_with_idx = []
        for image_file in all_image_files:
            idx_str = image_file.replace('gt_', '').replace('.png', '')
            try:
                frame_idx = int(idx_str)
                image_files_with_idx.append((frame_idx, image_file))
            except ValueError:
                continue
        if not image_files_with_idx:
            print(f"Warning: No valid gt_*.png in {comb_path}, skipping")
            continue
        image_files_with_idx.sort(key=lambda x: x[0])
        first_image_path = os.path.join(comb_path, image_files_with_idx[0][1])
        try:
            with Image.open(first_image_path) as img:
                image_width, image_height = img.size
        except Exception as e:
            print(f"Error reading image {first_image_path}: {e}, skipping")
            continue

        scene_name = f"{object_id}_{comb_folder}"
        output_metadata_dir = os.path.join(output_root, split, 'metadata')
        output_images_dir = os.path.join(output_root, split, 'images', scene_name)
        output_envmaps_dir = os.path.join(output_root, split, 'envmaps', scene_name)
        output_point_light_rays_dir = os.path.join(output_root, split, 'point_light_rays')
        output_pl_rays_path = os.path.join(output_point_light_rays_dir, f"{scene_name}.npy")
        os.makedirs(output_metadata_dir, exist_ok=True)

        comb_info = load_combined_light_info(comb_path)
        if comb_info is None:
            print(f"Warning: No combined.json in {comb_path}, skipping")
            continue

        has_envmap = comb_info['env_map'] is not None
        has_local_lights = len(comb_info['point_lights']) > 0 or len(comb_info['area_lights']) > 0

        # Determine skip logic
        skip_file_processing = False
        images_exist = os.path.exists(output_images_dir)
        envmap_files_exist = os.path.exists(output_envmaps_dir) if has_envmap else True
        rays_exist = os.path.exists(output_pl_rays_path) if has_local_lights else True
        if images_exist and envmap_files_exist and rays_exist:
            all_files_exist = all(
                os.path.exists(os.path.join(output_images_dir, f"{fi:05d}.png"))
                for fi, _ in image_files_with_idx
            )
            if all_files_exist:
                print(f"Skipping combined processing for {scene_name}: all files already exist")
                skip_file_processing = True

        os.makedirs(output_images_dir, exist_ok=True)
        if has_envmap:
            os.makedirs(output_envmaps_dir, exist_ok=True)
        if has_local_lights:
            os.makedirs(output_point_light_rays_dir, exist_ok=True)

        # Load environment map if present
        env_map_comb = None
        euler_rotation_comb = comb_info.get('rotation_euler', None)
        if not skip_file_processing and has_envmap and hdri_dir:
            env_map_name = comb_info['env_map']
            possible_names = [
                env_map_name, f"{env_map_name}.exr", f"{env_map_name}.hdr",
                f"{env_map_name}_8k.exr", f"{env_map_name}_8k.hdr",
            ]
            env_map_path = None
            for name in possible_names:
                test_path = os.path.join(hdri_dir, name)
                if os.path.exists(test_path):
                    env_map_path = test_path
                    break
            if env_map_path:
                env_map_comb = read_hdr(env_map_path)
                if env_map_comb is not None:
                    env_map_comb = torch.from_numpy(env_map_comb).float()
                else:
                    print(f"Warning: Failed to read envmap {env_map_path}")
            else:
                print(f"Warning: Environment map '{env_map_name}' not found in {hdri_dir}")

        light_area_weight_comb, view_dirs_comb = None, None
        if env_map_comb is not None:
            env_h, env_w = env_map_comb.shape[0], env_map_comb.shape[1]
            light_area_weight_comb, view_dirs_comb = generate_envir_map_dir(env_h, env_w)

        # Build point/area light rays if present
        if not skip_file_processing and has_local_lights:
            rays_arr = build_multi_source_light_rays(
                point_lights=comb_info['point_lights'],
                area_lights=comb_info['area_lights'],
                N=point_light_rays_n, scene_sphere_radius=scene_sphere_radius
            )
            np.save(output_pl_rays_path, rays_arr)

        # Process frames
        frames = []
        for frame_idx, image_file in image_files_with_idx:
            if frame_idx >= len(cameras_data):
                continue
            camera_info = cameras_data[frame_idx]
            c2w_blender = np.array(camera_info['c2w'])
            c2w_opencv = blender_to_opencv_c2w(c2w_blender)
            w2c = np.linalg.inv(c2w_opencv)
            fov = camera_info.get('fov', 30.0)
            fxfycxcy = fov_to_fxfycxcy(fov, image_width, image_height)
            output_image_name = f"{frame_idx:05d}.png"
            output_image_path = os.path.join(output_images_dir, output_image_name)

            if not skip_file_processing:
                shutil.copy2(os.path.join(comb_path, image_file), output_image_path)
                # Process envmap per frame if available
                if env_map_comb is not None:
                    env_hdr_raw, env_ldr, env_hdr = rotate_and_preprocess_envir_map(
                        env_map_comb, c2w_blender, euler_rotation=euler_rotation_comb,
                        light_area_weight=light_area_weight_comb, view_dirs=view_dirs_comb
                    )
                    if env_hdr_raw is not None:
                        env_hdr_uint8 = np.uint8(env_hdr * 255)
                        Image.fromarray(env_hdr_uint8).save(
                            os.path.join(output_envmaps_dir, f"{frame_idx:05d}_hdr.png"))
                        env_ldr_uint8 = np.uint8(env_ldr * 255)
                        Image.fromarray(env_ldr_uint8).save(
                            os.path.join(output_envmaps_dir, f"{frame_idx:05d}_ldr.png"))

            frames.append({
                "image_path": os.path.abspath(output_image_path),
                "fxfycxcy": fxfycxcy,
                "w2c": w2c.tolist()
            })

        scene_data = {"scene_name": scene_name, "frames": frames}
        with open(os.path.join(output_metadata_dir, f"{scene_name}.json"), 'w') as f:
            json.dump(scene_data, f, indent=2)

        desc = comb_info.get('description', '')
        extras = []
        if has_envmap:
            extras.append("envmap")
        if comb_info['point_lights']:
            extras.append(f"{len(comb_info['point_lights'])} point")
        if comb_info['area_lights']:
            extras.append(f"{len(comb_info['area_lights'])} area")
        print(f"Processed combined scene {scene_name}: {len(frames)} frames [{', '.join(extras)}] {desc}")
        processed_scene_names.append(scene_name)

    # Create tar for albedos folder (shared across all scenes with same object_id)
    # Only create once after processing all scenes for this object
    if output_tar_root is not None and output_albedos_dir and os.path.exists(output_albedos_dir):
        albedos_tar_path = os.path.join(output_tar_root, split, 'albedos', f"{object_id}.tar")
        albedos_tar_dir = albedos_tar_path.replace('.tar', '')
        # Check if there's an empty directory with the same name (without .tar)
        if os.path.exists(albedos_tar_dir) and os.path.isdir(albedos_tar_dir):
            try:
                if len(os.listdir(albedos_tar_dir)) == 0:
                    # Empty directory, remove it
                    print(f"Removing empty directory: {albedos_tar_dir}")
                    shutil.rmtree(albedos_tar_dir, ignore_errors=True)
                    if os.path.exists(albedos_tar_dir):
                        subprocess.run(['rm', '-rf', albedos_tar_dir], check=False)
            except Exception:
                pass
        if not os.path.exists(albedos_tar_path):
            create_tar_from_directory(output_albedos_dir, albedos_tar_path)

    # The public preprocessor deliberately preserves the raw renderer output.

    # Return list of all processed scene names (including skipped ones)
    # If no scenes were processed, return None
    if processed_scene_names:
        return processed_scene_names
    else:
        return None


def create_full_list(output_root, split='test', broken_scenes=None):
    """
    Create full_list.txt file listing all scene JSON files, excluding broken scenes.

    Args:
        output_root: Root directory for output
        split: 'train' or 'test'
        broken_scenes: List of broken scene names to exclude
    """
    metadata_dir = os.path.join(output_root, split, 'metadata')
    if not os.path.exists(metadata_dir):
        print(f"Warning: {metadata_dir} does not exist")
        return

    json_files = sorted([f for f in os.listdir(metadata_dir) if f.endswith('.json')])

    if broken_scenes is None:
        broken_scenes = []

    # Filter out broken scenes
    valid_json_files = []
    for json_file in json_files:
        scene_name = json_file.replace('.json', '')
        if scene_name not in broken_scenes:
            valid_json_files.append(json_file)

    full_list_path = os.path.join(output_root, split, 'full_list.txt')
    with open(full_list_path, 'w') as f:
        for json_file in valid_json_files:
            json_path = os.path.join(metadata_dir, json_file)
            # Write absolute path
            f.write(f"{os.path.abspath(json_path)}\n")

    print(f"Created {full_list_path} with {len(valid_json_files)} scenes (excluded {len(broken_scenes)} broken scenes)")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Objaverse data to re10k format')
    parser.add_argument('--input', '-i', default=None,
                       help='Input directory containing objaverse data (required unless --full-list-only)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for processed data (e.g., data_samples/objaverse_processed)')
    parser.add_argument('--output-tar', type=str, default=None,
                       help='Output directory for tar archives (location B). If omitted or empty, tar archiving is skipped.')
    parser.add_argument('--no-output-tar', action='store_true',
                       help='Do not create location B (no tar dirs, no tar copies, no B metadata). Overrides --output-tar.')
    parser.add_argument('--split', '-s', default='test', choices=['train', 'test'],
                       help='Split to process (default: test)')
    parser.add_argument('--object-id', type=str, default=None,
                       help='Process specific object ID only (default: process all)')
    parser.add_argument('--hdri-dir', type=str, default=None,
                       help='Directory containing HDR environment maps (e.g., data_samples/sample_hdris)')
    parser.add_argument('--test-run', action='store_true',
                       help='Test run: only process first 5 objects (default: False)')
    parser.add_argument('--max-objects', type=int, default=None,
                       help='Maximum number of objects to process (overrides --test-run if specified)')
    parser.add_argument('--point-light-rays-n', type=int, default=8192,
                       help='Number of rays per point-light scene (default: 8192)')
    parser.add_argument('--scene-sphere-radius', type=float, default=3.0,
                       help='Radius of scene bounding sphere for ray sampling (default: 3.0)')
    parser.add_argument('--full-list-only', action='store_true',
                       help='Skip all processing; only build full_list.txt from existing metadata in output.')

    args = parser.parse_args()

    output_root = args.output
    if args.no_output_tar:
        output_tar_root = None
    else:
        output_tar_root = args.output_tar
        if output_tar_root is not None and not str(output_tar_root).strip():
            output_tar_root = None

    if args.full_list_only:
        # Skip processing; only create full_list from existing metadata
        if not os.path.exists(output_root):
            print(f"Error: Output directory {output_root} does not exist")
            return
        broken_scenes = []
        broken_list_path = os.path.join(output_root, args.split, 'broken_scenes.txt')
        if os.path.exists(broken_list_path):
            with open(broken_list_path, 'r') as f:
                broken_scenes = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(broken_scenes)} broken scenes from {broken_list_path}")
        print(f"\nCreating full_list.txt for {args.split} split (full-list-only mode)...")
        create_full_list(output_root, split=args.split, broken_scenes=broken_scenes)
        if output_tar_root and os.path.exists(output_tar_root):
            create_full_list(output_tar_root, split=args.split, broken_scenes=broken_scenes)
            print(f"Created full_list.txt for location B")
        print("Done.")
        return

    if args.input is None:
        print("Error: --input is required when not using --full-list-only")
        return

    objaverse_root = args.input
    if not os.path.exists(objaverse_root):
        print(f"Error: Input directory {objaverse_root} does not exist")
        return

    # Find all object folders
    if args.object_id:
        object_ids = [args.object_id]
    else:
        object_ids = [d for d in os.listdir(objaverse_root)
                     if os.path.isdir(os.path.join(objaverse_root, d))]

    # Filter out objects without done.txt
    filtered_object_ids = []
    for object_id in object_ids:
        object_path = os.path.join(objaverse_root, object_id)
        done_file = os.path.join(object_path, 'done.txt')
        # if not os.path.exists(done_file):
        #     print(f"Skipping {object_id}: done.txt not found")
        #     continue
        filtered_object_ids.append(object_id)
    object_ids = filtered_object_ids

    # Apply test run or max objects limit
    original_count = len(object_ids)
    if args.max_objects is not None:
        object_ids = object_ids[:args.max_objects]
        print(f"Limiting to {args.max_objects} objects (from {original_count} total)")
    elif args.test_run:
        test_count = min(5, len(object_ids))
        object_ids = object_ids[:test_count]
        print(f"TEST RUN: Processing first {test_count} objects only (from {original_count} total)")

    print(f"Processing {len(object_ids)} objects...")

    broken_scenes = []
    processed_scenes = []
    skipped_scenes = []  # Scenes that were skipped because files already exist
    scenes_from_tar = []  # Scenes found in existing tar files

    for object_id in sorted(object_ids):
        print(f"\nProcessing object: {object_id}")
        object_path = os.path.join(objaverse_root, object_id)
        split_path = os.path.join(object_path, args.split)

        # Check for existing tar files in the input directory
        if os.path.exists(split_path):
            # Find tar files for env folders
            tar_files = [f for f in os.listdir(split_path)
                        if f.endswith('.tar') and (f.startswith('env_') or f.startswith('white_env_'))]
            pl_tar_files_main = [f for f in os.listdir(split_path)
                                 if f.endswith('.tar') and is_pl_folder(f.replace('.tar', ''))]

            # For each env tar file, check if corresponding scene exists in both output locations
            for tar_file in tar_files:
                env_folder = tar_file.replace('.tar', '')
                scene_name = f"{object_id}_{env_folder}"
                if check_scene_exists_in_outputs(scene_name, output_root, output_tar_root, args.split, is_point_light_scene=False):
                    if output_tar_root:
                        print(f"Scene {scene_name} already exists in both locations, adding to full_list")
                    else:
                        print(f"Scene {scene_name} already exists in location A, adding to full_list")
                    scenes_from_tar.append(scene_name)
                    # Ensure metadata JSON exists in location A
                    metadata_json = os.path.join(output_root, args.split, 'metadata', f"{scene_name}.json")
                    if not os.path.exists(metadata_json):
                        # Create a minimal JSON file if it doesn't exist
                        scene_data = {"scene_name": scene_name, "frames": []}
                        os.makedirs(os.path.dirname(metadata_json), exist_ok=True)
                        with open(metadata_json, 'w') as f:
                            json.dump(scene_data, f, indent=2)

                    # Ensure metadata JSON exists in location B with updated paths
                    if output_tar_root:
                        metadata_json_b = os.path.join(output_tar_root, args.split, 'metadata', f"{scene_name}.json")
                        if not os.path.exists(metadata_json_b):
                            # Load metadata from location A if it exists
                            if os.path.exists(metadata_json):
                                with open(metadata_json, 'r') as f:
                                    scene_data_a = json.load(f)
                            else:
                                scene_data_a = {"scene_name": scene_name, "frames": []}

                            # Create metadata for location B with updated image paths
                            images_tar_path = os.path.join(output_tar_root, args.split, 'images', f"{scene_name}.tar")
                            images_tar_abs_path = os.path.abspath(images_tar_path)

                            scene_data_b = {
                                "scene_name": scene_name,
                                "frames": []
                            }

                            for frame in scene_data_a.get('frames', []):
                                frame_b = frame.copy()
                                original_image_path = frame.get('image_path', '')
                                frame_filename = os.path.basename(original_image_path)
                                # Update path to point to location B tar file
                                frame_b['image_path'] = f"{images_tar_abs_path}::{frame_filename}"
                                scene_data_b['frames'].append(frame_b)

                            os.makedirs(os.path.dirname(metadata_json_b), exist_ok=True)
                            with open(metadata_json_b, 'w') as f:
                                json.dump(scene_data_b, f, indent=2)
                            print(f"Created metadata JSON for location B: {metadata_json_b}")

            # For each point-light tar file, check if scene exists and ensure metadata
            for tar_file in pl_tar_files_main:
                pl_folder = tar_file.replace('.tar', '')
                scene_name = f"{object_id}_{pl_folder}"
                if check_scene_exists_in_outputs(scene_name, output_root, output_tar_root, args.split, is_point_light_scene=True):
                    if output_tar_root:
                        print(f"Scene {scene_name} already exists in both locations, adding to full_list")
                    else:
                        print(f"Scene {scene_name} already exists in location A, adding to full_list")
                    scenes_from_tar.append(scene_name)
                    metadata_json = os.path.join(output_root, args.split, 'metadata', f"{scene_name}.json")
                    if not os.path.exists(metadata_json):
                        scene_data = {"scene_name": scene_name, "frames": []}
                        os.makedirs(os.path.dirname(metadata_json), exist_ok=True)
                        with open(metadata_json, 'w') as f:
                            json.dump(scene_data, f, indent=2)
                    if output_tar_root:
                        metadata_json_b = os.path.join(output_tar_root, args.split, 'metadata', f"{scene_name}.json")
                        if not os.path.exists(metadata_json_b):
                            if os.path.exists(metadata_json):
                                with open(metadata_json, 'r') as f:
                                    scene_data_a = json.load(f)
                            else:
                                scene_data_a = {"scene_name": scene_name, "frames": []}
                            images_tar_path = os.path.join(output_tar_root, args.split, 'images', f"{scene_name}.tar")
                            images_tar_abs_path = os.path.abspath(images_tar_path)
                            scene_data_b = {"scene_name": scene_name, "frames": []}
                            for frame in scene_data_a.get('frames', []):
                                frame_b = frame.copy()
                                frame_b['image_path'] = f"{images_tar_abs_path}::{os.path.basename(frame.get('image_path', ''))}"
                                scene_data_b['frames'].append(frame_b)
                            os.makedirs(os.path.dirname(metadata_json_b), exist_ok=True)
                            with open(metadata_json_b, 'w') as f:
                                json.dump(scene_data_b, f, indent=2)
                            print(f"Created metadata JSON for location B: {metadata_json_b}")

        try:
            result = process_objaverse_scene(objaverse_root, object_id, output_root, output_tar_root,
                                            split=args.split, hdri_dir=args.hdri_dir,
                                            point_light_rays_n=args.point_light_rays_n,
                                            scene_sphere_radius=args.scene_sphere_radius)
            if result == "broken":
                # Find all scenes for this object (env + point-light) and mark as broken
                split_path = os.path.join(objaverse_root, object_id, args.split)
                if os.path.exists(split_path):
                    for d in os.listdir(split_path):
                        item_path = os.path.join(split_path, d)
                        if os.path.isdir(item_path):
                            if d.startswith('env_') or d.startswith('white_env_') or is_pl_folder(d):
                                broken_scenes.append(f"{object_id}_{d}")
            elif isinstance(result, list):
                # result is a list of scene names (processed or skipped)
                # We can't distinguish between processed and skipped from the return value alone
                # But all scenes in the list should be added (they're either processed or skipped)
                for scene_name in result:
                    # Check if scene was skipped by checking if it exists in output
                    output_images_dir = os.path.join(output_root, args.split, 'images', scene_name)
                    output_json_path = os.path.join(output_root, args.split, 'metadata', f"{scene_name}.json")
                    if os.path.exists(output_images_dir) and os.path.exists(output_json_path):
                        # Check if files were just created (recent modification) or already existed
                        # For simplicity, we'll assume if it's in the result list, it was processed
                        # The skipped scenes are already in the output, so they'll be in full_list automatically
                        processed_scenes.append(scene_name)
            elif result is None:
                # No scenes were processed (maybe all were skipped or object had no valid scenes)
                pass
        except Exception as e:
            print(f"Error processing {object_id}: {e}")
            import traceback
            traceback.print_exc()

    # Add scenes from tar files to processed_scenes
    processed_scenes.extend(scenes_from_tar)

    # Create full_list.txt (excluding broken scenes)
    # Note: Skipped scenes are already in the metadata directory, so they will be included automatically
    print(f"\nCreating full_list.txt for {args.split} split...")
    create_full_list(output_root, split=args.split, broken_scenes=broken_scenes)

    # Also create full_list.txt in location B if it exists
    if output_tar_root:
        # Create full_list.txt for location B pointing to location B's metadata JSON files
        create_full_list(output_tar_root, split=args.split, broken_scenes=broken_scenes)
        print(f"Created full_list.txt for location B")

    # Save broken scenes list to a file
    if broken_scenes:
        broken_list_path = os.path.join(output_root, args.split, 'broken_scenes.txt')
        with open(broken_list_path, 'w') as f:
            for scene_name in sorted(broken_scenes):
                f.write(f"{scene_name}\n")
        print(f"Saved broken scenes list to {broken_list_path}")

    print(f"\nPreprocessing complete!")
    print(f"  - Processed {len(processed_scenes)} scenes")
    print(f"  - Skipped {len(skipped_scenes)} scenes (files already exist)")
    print(f"  - Marked {len(broken_scenes)} scenes as broken")


if __name__ == '__main__':
    main()
