"""Render scene-level LumiTokens training data.

Migrated from the LumiTokens data-generation repository and adapted to use
portable, command-line supplied dataset paths for the public release.
"""

import json
import math
import os
from dataclasses import dataclass
import random
from typing import Optional
import sys
import glob
import csv

# Ensure project root is on path so bpy_helper is found when run via Blender -b -P
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import imageio
import numpy as np
import simple_parsing
import shutil
import hashlib
error_list = []

@dataclass
class Options:
    """ 3D dataset rendering script """
    three_d_model_path: str = ''  # Set from glbs_root_path + CSV entry
    env_map_list_json: str = os.path.join(_script_dir, 'assets/hdri/polyhaven_hdris.json')
    env_map_dir_path: str = './data/hdri'
    white_env_map_dir_path: str = './data/hdri'
    output_dir: str = './outputs/rendered_scenes'
    num_views: int = 30  # Number of views
    num_test_views: int = 50  # Number of test views (trajectory views)
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
    csv_path: str = "object_ids.csv"  # Deprecated alias retained for compatibility
    rendered_dir_name: str = "rendered_scenes"  # Name of the rendered output directory (replaces 'glbs' in dataset path)
    texture_dir: str = "./data/polyhaven_textures"
    model_lq_dir: str = "./data/polyhaven_models"

    # New paths for curated lists
    lq_list_path: str = os.path.join(_script_dir, 'assets/object_ids/polyhaven_models_train.json')
    glb_list_path: str = 'object_ids.csv'
    glbs_root_path: str = './data/objaverse/glbs'

    # Random seed for scene composition
    scene_seed: Optional[int] = None
    render_resolution: int = 512
    cycles_samples: int = 64
    use_denoising: bool = True


def render_core(args: Options, groups_id = 0):
    import bpy
    import mathutils
    from mathutils import Matrix

    from bpy_helper.camera import create_camera, look_at_to_c2w
    from bpy_helper.io import render_depth_map, mat2list, array2list, render_normal_map, render_albedo_map, transform_normals_to_camera_space
    from bpy_helper.light import create_point_light, set_env_light, create_area_light
    from bpy_helper.material import create_white_diffuse_material, create_specular_ggx_material, clear_emission_and_alpha_nodes
    from bpy_helper.random import gen_random_pts_around_origin, gen_pt_traj_around_origin
    from bpy_helper.scene import import_3d_model, normalize_scene, reset_scene
    from bpy_helper.utils import stdout_redirected

    def add_textured_plane(texture_dir):
        # Create a large plane
        size = random.uniform(30.0, 50.0)
        bpy.ops.mesh.primitive_plane_add(size=size)
        plane = bpy.context.active_object
        plane.name = "GroundPlane"

        # Rotate arbitrarily around up axis
        plane.rotation_euler = [0, 0, random.uniform(0, 2*math.pi)]

        # 50% chance to apply texture, 50% chance to leave it without texture
        apply_texture = random.random() < 0.5

        if apply_texture and os.path.exists(texture_dir):
            try:
                subdirs = [d for d in os.listdir(texture_dir) if os.path.isdir(os.path.join(texture_dir, d))]
                if subdirs:
                    chosen_subdir = random.choice(subdirs)
                    search_path = os.path.join(texture_dir, chosen_subdir)

                    candidates = []
                    for root, dirs, files in os.walk(search_path):
                        for file in files:
                            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.exr')):
                                candidates.append(os.path.join(root, file))

                    diff_candidates = [f for f in candidates if any(k in f.lower() for k in ['diff', 'col', 'albedo'])]

                    if diff_candidates:
                        texture_path = random.choice(diff_candidates)

                        mat = bpy.data.materials.new(name="PlaneMaterial")
                        mat.use_nodes = True
                        nodes = mat.node_tree.nodes
                        links = mat.node_tree.links
                        bsdf = nodes.get("Principled BSDF")

                        tex_image = nodes.new('ShaderNodeTexImage')
                        try:
                            img = bpy.data.images.load(texture_path)
                            tex_image.image = img
                            links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
                            print(f"Applied texture to plane: {texture_path}")
                        except Exception as e:
                            print(f"Could not load texture {texture_path}: {e}")

                        if plane.data.materials:
                            plane.data.materials[0] = mat
                        else:
                            plane.data.materials.append(mat)

                        bpy.ops.object.mode_set(mode='EDIT')
                        bpy.ops.uv.smart_project()
                        bpy.ops.object.mode_set(mode='OBJECT')
            except Exception as e:
                print(f"Error applying texture to plane: {e}")
        else:
            print(f"Ground plane created without texture (apply_texture={apply_texture})")

    def get_world_bbox(obj):
        """
        Calculate the world-space bounding box of an object.
        Returns (min_corner, max_corner) as Vectors.
        """
        bbox_min = mathutils.Vector((math.inf, math.inf, math.inf))
        bbox_max = mathutils.Vector((-math.inf, -math.inf, -math.inf))

        # Ensure we use the evaluated object to get correct world transforms
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)

        if eval_obj.type == 'MESH':
            # Transform all 8 corners of the local bbox to world space
            for corner in eval_obj.bound_box:
                world_corner = eval_obj.matrix_world @ mathutils.Vector(corner)
                bbox_min.x = min(bbox_min.x, world_corner.x)
                bbox_min.y = min(bbox_min.y, world_corner.y)
                bbox_min.z = min(bbox_min.z, world_corner.z)
                bbox_max.x = max(bbox_max.x, world_corner.x)
                bbox_max.y = max(bbox_max.y, world_corner.y)
                bbox_max.z = max(bbox_max.z, world_corner.z)

        return bbox_min, bbox_max

    def check_collision(obj1, obj2):
        """
        Check if two objects are colliding using Axis-Aligned Bounding Box (AABB) overlap.
        Returns True if they overlap, False otherwise.
        """
        try:
            min1, max1 = get_world_bbox(obj1)
            min2, max2 = get_world_bbox(obj2)

            # Check for overlap in all 3 dimensions
            # If there is a gap in ANY dimension, there is no collision
            overlap_x = (min1.x <= max2.x) and (max1.x >= min2.x)
            overlap_y = (min1.y <= max2.y) and (max1.y >= min2.y)
            overlap_z = (min1.z <= max2.z) and (max1.z >= min2.z)

            return overlap_x and overlap_y and overlap_z

        except Exception as e:
            # print(f"AABB collision check failed: {e}")
            return False

    def place_object_randomly(model_objects, existing_objects, scale_range=(0.5, 1.0)):
        if not model_objects:
            return

        # 1. Scale
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found_mesh = False

        # Ensure we are in object mode
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        for obj in model_objects:
            if obj.type == 'MESH':
                found_mesh = True
                for coord in obj.bound_box:
                    coord = mathutils.Vector(coord)
                    coord = obj.matrix_world @ coord
                    bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                    bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

        if found_mesh:
            max_dim = max(bbox_max[i] - bbox_min[i] for i in range(3))
            target_scale = random.uniform(*scale_range)
            scale_factor = target_scale / max_dim if max_dim > 0 else 1.0

            for obj in model_objects:
                if obj.parent is None:
                    obj.scale = obj.scale * scale_factor
            bpy.context.view_layer.update()

            # 2. Center and move to ground
            bbox_min = (math.inf,) * 3
            bbox_max = (-math.inf,) * 3
            for obj in model_objects:
                if obj.type == 'MESH':
                    for coord in obj.bound_box:
                        coord = mathutils.Vector(coord)
                        coord = obj.matrix_world @ coord
                        bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                        bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

            current_center = (mathutils.Vector(bbox_min) + mathutils.Vector(bbox_max)) / 2
            centering_offset = -current_center

            for obj in model_objects:
                if obj.parent is None:
                    obj.matrix_world.translation += centering_offset
            bpy.context.view_layer.update()

            # Move to ground (z=0)
            bbox_min_z = math.inf
            for obj in model_objects:
                if obj.type == 'MESH':
                    for coord in obj.bound_box:
                        coord = mathutils.Vector(coord)
                        coord = obj.matrix_world @ coord
                        if coord.z < bbox_min_z:
                            bbox_min_z = coord.z

            ground_offset = mathutils.Vector((0, 0, -bbox_min_z))
            for obj in model_objects:
                if obj.parent is None:
                    obj.matrix_world.translation += ground_offset
            bpy.context.view_layer.update()

            # 2.5 Random Rotation around Z-axis
            # Apply random rotation to the root object(s)
            random_angle = random.uniform(0, 2 * math.pi)
            rot_matrix = mathutils.Matrix.Rotation(random_angle, 4, 'Z')
            print(f"DEBUG: Random rotation angle: {random_angle}")

            # We need to rotate around the object's current center (which is now at 0,0,0 in XY)
            # Since we centered it, applying rotation to the world matrix should work fine if pivot is origin
            # or just rotate the object itself.
            for obj in model_objects:
                if obj.parent is None:
                    # Apply rotation. Since we are at origin (mostly), this rotates around Z axis at origin.
                    # If the object was moved to ground, its Z is not 0, but X,Y are 0.
                    # So rotation around Z axis (0,0,1) passing through origin is correct.
                    obj.matrix_world = rot_matrix @ obj.matrix_world
            bpy.context.view_layer.update()

            # 3. Position avoiding collision
            valid_pos_found = False
            original_locations = {obj: obj.location.copy() for obj in model_objects if obj.parent is None}

            for _ in range(1000):
                dist = random.uniform(0, 1.5)
                theta = random.uniform(0, 2*math.pi)
                x = dist * math.cos(theta)
                y = dist * math.sin(theta)
                z = 0
                offset = mathutils.Vector((x, y, z))

                for obj in model_objects:
                    if obj.parent is None:
                        obj.location = original_locations[obj] + offset
                bpy.context.view_layer.update()

                is_colliding = False
                for other_obj in existing_objects:
                    # Skip GroundPlane for collision checks
                    if other_obj.name == "GroundPlane":
                        continue

                    if other_obj.type == 'MESH':
                         for model_obj in model_objects:
                             if model_obj.type == 'MESH':
                                 if check_collision(model_obj, other_obj):
                                     is_colliding = True
                                     print(f"collision found")
                                     break
                    if is_colliding:
                        break

                if not is_colliding:
                    valid_pos_found = True
                    print(f"DEBUG: Found valid position at offset {offset}")
                    break

            if not valid_pos_found:
                 print("DEBUG: Could not find collision-free position. Moving to fallback.")
                 fallback_offset = mathutils.Vector((-10.0, 0, 0)) # Move far away if fails
                 for obj in model_objects:
                    if obj.parent is None:
                        obj.location = original_locations[obj] + fallback_offset
                 bpy.context.view_layer.update()

    def add_lq_model(model_dir, lq_candidates, existing_objects=[]):
        if not os.path.exists(model_dir):
            print(f"LQ Model dir {model_dir} does not exist.")
            return []

        if not lq_candidates:
            print("No LQ candidates provided")
            return []

        # Pick one random LQ model
        model_id = random.choice(lq_candidates)

        # Search for .blend files in the model directory
        # Structure is typically model_dir/model_id/resolution/model_id_res.blend
        search_path = os.path.join(model_dir, model_id, "**", "*.blend")
        files = glob.glob(search_path, recursive=True)

        if not files:
            print(f"Could not find .blend file for {model_id} in {os.path.join(model_dir, model_id)}")
            return []

        # Prefer 4k, then 1k, then whatever
        filepath = None
        for f in files:
            if "4k" in f:
                filepath = f
                break
        if not filepath:
            for f in files:
                if "1k" in f:
                    filepath = f
                    break
        if not filepath:
            filepath = files[0]

        print(f"Loading LQ model: {model_id} from {filepath}")

        # Load objects from .blend file
        with bpy.data.libraries.load(filepath) as (data_from, data_to):
            data_to.objects = data_from.objects

        lq_objects = []
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
                lq_objects.append(obj)

        if not lq_objects:
            print("No objects loaded from .blend file")
            return []

        place_object_randomly(lq_objects, existing_objects, scale_range=(0.8, 1.2))
        return lq_objects

    def add_glb_model(filepath, existing_objects=[], scale_range=(0.7, 1.0)):
        if not os.path.exists(filepath):
            print(f"GLB file not found: {filepath}")
            return []

        # Capture objects before import
        objs_before = set(bpy.context.scene.objects)

        with stdout_redirected():
            import_3d_model(filepath)

        objs_after = set(bpy.context.scene.objects)
        new_objects = list(objs_after - objs_before)

        if new_objects:
            place_object_randomly(new_objects, existing_objects, scale_range=scale_range)

        return new_objects

    def render_rgb_and_hint(output_path,idx = 0):
        # Get the last added object (assuming the new object is the most recently added one)
        # new_object = bpy.context.scene.objects[-1]
        # Set the name for the newly imported object
        # new_object.name = "shape"
        # bpy.context.view_layer.objects.active = new_object
        # bpy.context.view_layer.update()

        bpy.context.scene.view_layers["ViewLayer"].material_override = None
        bpy.context.scene.render.image_settings.file_format = 'PNG'  # set output to png (with tonemapping)
        bpy.context.scene.render.filepath = os.path.join(output_path, f'gt_{idx}.png')

        # with stdout_redirected():
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        img = imageio.v3.imread(os.path.join(output_path, f'gt_{idx}.png')) / 255.
        if img.shape[-1] == 4:
            img = img[..., :3] * img[..., 3:]  # fix edge aliasing
        imageio.v3.imwrite(
            os.path.join(output_path, f'gt_{idx}.png'),
            (img * 255).clip(0, 255).astype(np.uint8),
        )

    def configure_blender():
        # Set the render resolution
        bpy.context.scene.render.resolution_x = args.render_resolution
        bpy.context.scene.render.resolution_y = args.render_resolution
        bpy.context.scene.render.resolution_percentage = 100
        # 在 configure_blender() 中，cycles.device = 'GPU' 之后添加：
        bpy.context.scene.cycles.tile_x = 256
        bpy.context.scene.cycles.tile_y = 256
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.preferences.addons["cycles"].preferences.get_devices()

        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.samples = args.cycles_samples
        bpy.context.scene.cycles.use_denoising = args.use_denoising
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

        # Enable the alpha channel for GT mask
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    reset_scene()

    #& 1.preparing the scene
    #* 1.1 prepare the 3d model

    # Set scene seed if provided
    if args.scene_seed is not None:
        # We use a combination of scene_seed and groups_id to vary per group if needed,
        # or just set it once. Let's make it deterministic per group.
        # compute the output_dir str into a seed integer based on the hash of the string:
        current_seed = int(hashlib.sha256(args.output_dir.encode()).hexdigest(), 16) % 1000000

    # Add ground plane first
    add_textured_plane(args.texture_dir)

    # Track all objects in the scene to avoid collisions
    # Start with ground plane
    existing_objects = [obj for obj in bpy.context.scene.objects if obj.name == "GroundPlane"]

    # --- Load LQ Object (1) ---
    # Load curated LQ list
    lq_candidates = []
    if os.path.exists(args.lq_list_path):
        try:
            with open(args.lq_list_path, 'r') as f:
                lq_candidates = json.load(f)
        except Exception as e:
            print(f"Error loading LQ list: {e}")

    if lq_candidates:
        print("Loading 1 LQ object")
        new_objs = add_lq_model(args.model_lq_dir, lq_candidates, existing_objects)
        if new_objs:
            existing_objects.extend(new_objs)

    # --- Load Additional GLB Objects (0-7) ---
    # Load curated GLB list
    glb_candidates = []
    if os.path.exists(args.glb_list_path):
        try:
            with open(args.glb_list_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        glb_candidates.append((row[0].strip(), row[1].strip()))
        except Exception as e:
            print(f"Error loading GLB list: {e}")

    if glb_candidates:
        num_glbs = random.randint(2, 7)
        print(f"Loading {num_glbs} additional GLB objects")
        for _ in range(num_glbs):
            idx, uid = random.choice(glb_candidates)
            glb_path = os.path.join(args.glbs_root_path, idx, f"{uid}.glb")
            new_objs = add_glb_model(glb_path, existing_objects, scale_range=(0.5, 1.0))
            if new_objs:
                existing_objects.extend(new_objs)

    # Debug: Print all objects final locations
    for obj in bpy.context.scene.objects:
        print(f"DEBUG: Object: {obj.name}, Location: {obj.location}, Scale: {obj.scale}")

    # scale, offset = normalize_scene(use_bounding_sphere=True)
    # Instead of normalizing the whole scene (which would rescale everything again),
    # we just calculate the overall scene scale/offset for saving metadata, if needed.
    # But since we manually placed them, we can just set dummy values or calculate actuals.
    scale = 1.0
    offset = [0.0, 0.0, 0.0]

    clear_emission_and_alpha_nodes()

    # Configure blender
    configure_blender()

    # Load env map list
    env_map_list = json.load(open(args.env_map_list_json, 'r'))

    # Render GT images & hints
    seed_view = None if args.seed is None else args.seed
    seed_white_pl = None if args.seed is None else args.seed + 1
    seed_rgb_pl = None if args.seed is None else args.seed + 2
    seed_multi_pl = None if args.seed is None else args.seed + 3
    seed_area = None if args.seed is None else args.seed + 4
    seed_combined = None if args.seed is None else args.seed + 5
    # file_path is not defined here, we should use args.three_d_model_path or extract the UID
    # The original code used file_path which was args.three_d_model_path
    file_path = args.three_d_model_path
    res_dir = f"{args.output_dir}/{file_path.split('/')[-1].split('.')[0]}"
    os.makedirs(res_dir, exist_ok=True)
    res_dir = os.path.join(res_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    json.dump({'scale': scale, 'offset': array2list(offset)}, open(f'{res_dir}/normalize.json', 'w'), indent=4)

    #* 1.2 prepare the cameras
    # Check if cameras.json exists in train and test folders
    train_cam_path = os.path.join(res_dir, 'train', 'cameras.json')
    test_cam_path = os.path.join(res_dir, 'test', 'cameras.json')

    cameras = []
    cameras_test = []
    loaded_existing_cameras = False

    if os.path.exists(train_cam_path) and os.path.exists(test_cam_path):
        print(f"Loading existing cameras from {res_dir}")
        try:
            with open(train_cam_path, 'r') as f:
                train_cams_data = json.load(f)
            for cam in train_cams_data:
                # Convert list back to Matrix if needed, but create_camera usually handles it.
                # However, to be safe and consistent with generation, we keep it as list or convert to Matrix.
                # bpy_helper.camera.create_camera likely expects Matrix or compatible.
                # Let's convert to Matrix to be safe.
                c2w = Matrix(cam['c2w'])
                cameras.append((cam['eye_idx'], c2w, cam['fov']))

            with open(test_cam_path, 'r') as f:
                test_cams_data = json.load(f)
            for cam in test_cams_data:
                c2w = Matrix(cam['c2w'])
                cameras_test.append((cam['eye_idx'], c2w, cam['fov']))

            loaded_existing_cameras = True
        except Exception as e:
            print(f"Failed to load existing cameras: {e}. Generating new ones.")
            loaded_existing_cameras = False

    if not loaded_existing_cameras:
        eyes = gen_random_pts_around_origin(
            seed=seed_view,
            N=args.num_views,                # set to a large value (e.g. 100, 200, 400)
            min_dist_to_origin=2.5,
            max_dist_to_origin=3.5,          # usually keep min=max for consistent radius
            min_theta_in_degree=20,           # 0 for full sphere, 10/20 for hemisphere
            max_theta_in_degree=80,         # 90 or 70 for upper hemisphere only
            z_up=True
        )
        eyes_traj = gen_pt_traj_around_origin(
            seed=seed_view,
            N=args.num_test_views,
            min_dist_to_origin=3.0,
            max_dist_to_origin=3.0,
            theta_in_degree=60,
            z_up=True
        )

        cameras = []
        cameras_test = []
        for eye_idx, eye in enumerate(eyes):
            fov = 30
            radius = (0.5 / math.tanh(fov / 2. * (math.pi / 180.)))
            eye = [x * radius for x in eye]
            c2w = look_at_to_c2w(eye)
            cameras.append((eye_idx, c2w, fov))

        for eye_idx, eye in enumerate(eyes_traj):
            fov = 30
            radius = (0.5 / math.tanh(fov / 2. * (math.pi / 180.)))
            eye = [x * radius for x in eye]
            c2w = look_at_to_c2w(eye)
            cameras_test.append((eye_idx, c2w, fov))

    #& 2. start rendering
    # If we loaded existing cameras, we assume intrinsics might be done, but let's be careful.
    # The user said: "if at least one lighting variation is already rendered, we should not be rendering the intrinsics again"
    # If loaded_existing_cameras is True, it means we have cameras.json, which implies at least one light render pass finished.
    if loaded_existing_cameras:
        intrinsics_saved = True
    else:
        intrinsics_saved = not args.save_intrinsics

    def is_folder_populated(path, num_expected):
        # Check if tar file exists
        if os.path.exists(path + '.tar'):
            return True

        if not os.path.exists(path):
            return False
        # Check for png files. User mentioned n+1, but we check for at least n to be safe.
        png_files = glob.glob(os.path.join(path, '*.png'))
        return len(png_files) >= num_expected

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
        strength = 1.0
        set_env_light(env_map_path, rotation_euler=rotation_euler, strength=strength)

        all_cams = []  # <-- Collect all cams for this env

        for eye_idx, c2w, fov in cameras:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/train'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            if not intrinsics_saved:
                with stdout_redirected():
                    render_depth_map(view_path, file_prefix=f'depth_{eye_idx}')
                    render_normal_map(view_path)
                    render_albedo_map(view_path)
                # copy the depth map to a different name
                depth_folder = os.path.join(view_path, 'depth')
                os.makedirs(depth_folder, exist_ok=True)
                depth_path = os.path.join(view_path, f'depth_{eye_idx}0001.exr')
                depth_cam_path = os.path.join(depth_folder, f'depth_{eye_idx}.exr')
                shutil.copy(depth_path, depth_cam_path)
                # Transform normals to camera space
                normals_path = os.path.join(view_path, 'normal0001.exr')
                normal_folder = os.path.join(view_path, 'normal')
                os.makedirs(normal_folder, exist_ok=True)
                normals_cam_path = os.path.join(normal_folder, f'normal_cam_{eye_idx}.exr')
                transform_normals_to_camera_space(normals_path, c2w, normals_cam_path)
                albedo_path = os.path.join(view_path, 'albedo0001.png')
                albedo_folder = os.path.join(view_path, 'albedo')
                os.makedirs(albedo_folder, exist_ok=True)
                albedo_cam_path = os.path.join(albedo_folder, f'albedo_cam_{eye_idx}.png')
                shutil.copy(albedo_path, albedo_cam_path)
                # clean up the files before they got moved:
                os.remove(os.path.join(view_path, f'depth_{eye_idx}0001.exr'))
                os.remove(os.path.join(view_path, 'normal0001.exr'))
                os.remove(os.path.join(view_path, 'albedo0001.png'))
                # remove ant files with "rgb_for_" prefix
                for file in os.listdir(view_path):
                    if file.startswith('rgb_for_'):
                        os.remove(os.path.join(view_path, file))

            # Instead of saving cam.json per view, collect the info:
            cam_entry = {
                'eye_idx': eye_idx,
                'c2w': mat2list(c2w),
                'fov': fov,
            }
            all_cams.append(cam_entry)

            env_path = f'{view_path}/white_env_{env_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        # === Save all camera info for this env in a single file ===
        cameras_json_path = os.path.join(view_path, f'cameras.json')
        json.dump(all_cams, open(cameras_json_path, 'w'), indent=4)

        #* render the test views for white env lighting
        all_cams_test = []
        for eye_idx, c2w, fov in cameras_test:
            camera = create_camera(c2w, fov)
            bpy.context.scene.camera = camera
            view_path = f'{res_dir}/test'
            if not os.path.exists(view_path):
                os.makedirs(view_path)

            if not intrinsics_saved:
                with stdout_redirected():
                    render_depth_map(view_path, file_prefix=f'depth_{eye_idx}')
                    render_normal_map(view_path)
                    render_albedo_map(view_path)
                # copy the depth map to a different name
                depth_folder = os.path.join(view_path, 'depth')
                os.makedirs(depth_folder, exist_ok=True)
                depth_path = os.path.join(view_path, f'depth_{eye_idx}0001.exr')
                depth_cam_path = os.path.join(depth_folder, f'depth_{eye_idx}.exr')
                shutil.copy(depth_path, depth_cam_path)
                # Transform normals to camera space
                normals_path = os.path.join(view_path, 'normal0001.exr')
                normal_folder = os.path.join(view_path, 'normal')
                os.makedirs(normal_folder, exist_ok=True)
                normals_cam_path = os.path.join(normal_folder, f'normal_cam_{eye_idx}.exr')
                transform_normals_to_camera_space(normals_path, c2w, normals_cam_path)
                albedo_path = os.path.join(view_path, 'albedo0001.png')
                albedo_folder = os.path.join(view_path, 'albedo')
                os.makedirs(albedo_folder, exist_ok=True)
                albedo_cam_path = os.path.join(albedo_folder, f'albedo_cam_{eye_idx}.png')
                shutil.copy(albedo_path, albedo_cam_path)
                # clean up the files before they got moved:
                os.remove(os.path.join(view_path, f'depth_{eye_idx}0001.exr'))
                os.remove(os.path.join(view_path, 'normal0001.exr'))
                os.remove(os.path.join(view_path, 'albedo0001.png'))
                # remove ant files with "rgb_for_" prefix
                for file in os.listdir(view_path):
                    if file.startswith('rgb_for_'):
                        os.remove(os.path.join(view_path, file))

            # Instead of saving cam.json per view, collect the info:
            cam_entry = {
                'eye_idx': eye_idx,
                'c2w': mat2list(c2w),
                'fov': fov,
            }
            all_cams_test.append(cam_entry)

            env_path = f'{view_path}/white_env_{env_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}', eye_idx)

            bpy.data.objects.remove(camera, do_unlink=True)

        # === Save all camera info for this env in a single file ===
        cameras_json_path = os.path.join(view_path, f'cameras.json')
        json.dump(all_cams_test, open(cameras_json_path, 'w'), indent=4)

        # save the env map
        json.dump({
            'env_map': 'white_env_8k.exr',
            'rotation_euler': rotation_euler,
            'strength': strength,
            }, open(f'{env_path}/white_env.json', 'w'), indent=4)

        intrinsics_saved = True

    #* 2.2 render the white point lighting
    white_pls = gen_random_pts_around_origin(
        seed=seed_white_pl,
        N=args.num_white_pls,
        min_dist_to_origin=3.5,
        max_dist_to_origin=5.0,
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
        power = random.uniform(500, 1500)
        _point_light = create_point_light(pl, power)

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
        json.dump({
            'pos': array2list(pl),
            'power': power,
        }, open(f'{env_path}/white_pl.json', 'w'), indent=4)

    #* 2.3 render the RGB point lighting
    rgb_pls = gen_random_pts_around_origin(
        seed=seed_rgb_pl,
        N=args.num_rgb_pls,
        min_dist_to_origin=4.0,
        max_dist_to_origin=5.0,
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
        power = random.uniform(900, 1500)  # slightly brighter than white light
        rgb = [random.uniform(0, 1) for _ in range(3)]
        create_point_light(pl, power, rgb=rgb)

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
        json.dump({
            'pos': array2list(pl),
            'power': power,
            'color': rgb,
        }, open(f'{env_path}/rgb_pl.json', 'w'), indent=4)

    #* 2.4 render the multi point lighting
    multi_pls = gen_random_pts_around_origin(
        seed=seed_multi_pl,
        N=args.num_multi_pls * args.max_pl_num,
        min_dist_to_origin=3.0,
        max_dist_to_origin=5.0,
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
        powers = [random.uniform(500, 1500) for _ in range(args.max_pl_num)]
        colors = []
        for pl_idx in range(args.max_pl_num):
            if random.random() < 0.5:
                rgb = [1.0, 1.0, 1.0]  # white
            else:
                rgb = [random.uniform(0.4, 1.0) for _ in range(3)]  # colored
            colors.append(rgb)
            create_point_light(pls[pl_idx], powers[pl_idx], rgb=rgb, keep_other_lights=pl_idx > 0)

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
        json.dump({
            'pos': mat2list(pls),
            'power': powers,
            'color': colors,
        }, open(f'{env_path}/multi_pl.json', 'w'), indent=4)

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
        strength = 1.0
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
        json.dump({
            'env_map': env_map,
            'rotation_euler': rotation_euler,
            'strength': strength,
        }, open(f'{env_path}/env.json', 'w'), indent=4)

    #* 2.6 render the area lighting
    area_light_positions = gen_random_pts_around_origin(
        seed=seed_area,
        N=args.num_area_lights,
        min_dist_to_origin=3.0,
        max_dist_to_origin=6.0,
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
        area_light_power = random.uniform(700, 1500)
        area_light_size = random.uniform(5., 10.)
        if random.random() < 0.75:
            color = [1.0, 1.0, 1.0]  # white
        else:
            color = [random.uniform(0.4, 1.0) for _ in range(3)]  # colored

        _area_light = create_area_light(area_light_pos, area_light_power, area_light_size, color=color)

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
        json.dump({
            'pos': array2list(area_light_pos),
            'power': area_light_power,
            'size': area_light_size,
            'color': color,
        }, open(f'{env_path}/area.json', 'w'), indent=4)

    #* 2.7 render the combined lighting (progressive: env -> +point1 -> +point2 -> +area)
    # Generate positions for point lights and area light
    num_point_lights = min(2, args.max_pl_num)  # Use up to 2 point lights
    combined_pls = gen_random_pts_around_origin(
        seed=seed_combined,
        N=num_point_lights,
        min_dist_to_origin=3.5,
        max_dist_to_origin=5.0,
        min_theta_in_degree=0,
        max_theta_in_degree=85
    )
    area_light_positions = gen_random_pts_around_origin(
        seed=seed_combined + 100 if seed_combined is not None else None,
        N=1,
        min_dist_to_origin=3.0,
        max_dist_to_origin=6.0,
        min_theta_in_degree=0,
        max_theta_in_degree=85
    )

    if args.num_combined_lights > 0:
        # Setup: Choose env map parameters (shared across all progressive stages)
        env_map = random.choice(env_map_list)
        env_map_path = f'{args.env_map_dir_path}/{env_map}_8k.exr'
        rotation_euler = [0, 0, random.uniform(-math.pi, math.pi)]
        strength = 1.0

        # Generate light parameters
        point_powers = [random.uniform(500, 1500) for _ in range(num_point_lights)]
        point_colors = [[1.0, 1.0, 1.0] for _ in range(num_point_lights)]

        area_light_pos = area_light_positions[0]
        area_light_power = random.uniform(700, 1500)
        area_light_size = random.uniform(5., 10.)
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
                    create_point_light(combined_pls[pl_idx], point_powers[pl_idx], rgb=point_colors[pl_idx], keep_other_lights=(pl_idx > 0 or True))
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
            train_json_path = f'{res_dir}/train/combined_{stage_idx}'
            json.dump(light_info, open(f'{train_json_path}/combined.json', 'w'), indent=4)

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
            test_json_path = f'{res_dir}/test/combined_{stage_idx}'
            json.dump(light_info, open(f'{test_json_path}/combined.json', 'w'), indent=4)

    # store a file indicating the end of the rendering
    with open(os.path.join(res_dir, 'done.txt'), 'w') as f:
        f.write('done')
        f.close()


if __name__ == '__main__':
    # When run via Blender -b -P script.py -- args, we need to parse only args after --
    # sys.argv includes Blender's own arguments before --
    import sys
    if '--' in sys.argv:
        script_args = sys.argv[sys.argv.index('--') + 1:]
    else:
        script_args = sys.argv[1:]

    args: Options = simple_parsing.parse(Options, args=script_args)
    if args.scene_seed is not None:
        random.seed(args.scene_seed)
        np.random.seed(args.scene_seed)
        print(f"Setting scene random seed to {args.scene_seed}")
    dataset_path = os.path.abspath(args.glbs_root_path)
    args.glbs_root_path = dataset_path
    csv_file = args.glb_list_path
    args.output_dir = os.path.abspath(args.output_dir)

    print(Options)
    import csv
    index_uid_list = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                index, uid = row
                index_uid_list.append((index.strip(), uid.strip()))
    # Preview
    print(f"Loaded {len(index_uid_list)} entries")
    if not index_uid_list:
        raise ValueError(f"No 'shard,uid' rows found in {csv_file}")
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
        # Set the seed for reproducibility
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
        # Render the model
        print('Rendering model:', uid)
        if uid in error_list:
            print('skipping this model')
            continue
        for j in range(args.num_view_groups):
            # if found a done.txt file, skip this model
            print('rendering group:', j)
            target_dir = os.path.join(args.output_dir, uid)
            if os.path.exists(os.path.join(target_dir, 'done.txt')):
                print(f"Skipping {uid} (done.txt found)")
                continue

            # If not done, but directory exists, remove it to start fresh
            if os.path.exists(target_dir):
                print(f"Removing incomplete directory: {target_dir}")
                shutil.rmtree(target_dir)

            render_core(args, j)
            print('render progress:', i, 'of range', args.group_start, '~', args.group_end)
