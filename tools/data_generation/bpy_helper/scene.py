import math

import bpy
import mathutils


def remove_default_objects() -> None:
    """
    Removes the default objects from the scene.
    """

    objects_to_remove = ["Cube", "Camera", "Light"]

    for obj_name in objects_to_remove:
        bpy.ops.object.select_all(action='DESELECT')
        obj = bpy.data.objects.get(obj_name)
        if obj is not None:
            obj.select_set(True)
            bpy.ops.object.delete()


def reset_scene() -> None:
    """
    Resets the scene to the default state.
    """

    bpy.ops.wm.read_homefile()

    # Remove the default cube
    remove_default_objects()


def import_3d_model(object_path) -> None:
    """
    Loads a 3d model into the scene.
    """

    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=object_path)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=object_path)
    elif object_path.endswith(".ply"):
        bpy.ops.import_mesh.ply(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_root_objects():
    """
    Yields all root objects in the scene.
    """

    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    """
    Yields all mesh objects in the scene.
    """

    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            yield obj


def scene_bbox(single_obj=None, ignore_matrix=False) -> tuple[mathutils.Vector, mathutils.Vector]:
    """
    Compute the bounding box of the scene.

    :param single_obj: if not None, only compute the bounding box for this object
    :param ignore_matrix: if True, ignore the transformation matrix of the object
    :return: bounding box min and max
    """

    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = mathutils.Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return mathutils.Vector(bbox_min), mathutils.Vector(bbox_max)


def get_center(l) -> float:
    """
    Compute the center value of a list of numbers.
    """

    return (max(l) + min(l)) / 2 if l else 0.0


def scene_sphere(single_obj=None, ignore_matrix=False) -> tuple[mathutils.Vector, float]:
    """
    Compute the bounding sphere of the scene.

    :param single_obj: if not None, only compute the bounding sphere for this object
    :param ignore_matrix: if True, ignore the transformation matrix of the object
    :return: bounding sphere center and radius
    """

    found = False
    points_co_global = []
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        mesh = obj.data
        for vertex in mesh.vertices:
            vertex_co = vertex.co
            if not ignore_matrix:
                vertex_co = obj.matrix_world @ vertex_co
            points_co_global.extend([vertex_co])
    if not found:
        raise RuntimeError("no objects in scene to compute bounding sphere for")
    x, y, z = [[point_co[i] for point_co in points_co_global] for i in range(3)]
    b_sphere_center = mathutils.Vector([get_center(axis) for axis in [x, y, z]]) if (x and y and z) else None
    b_sphere_radius = max(
        (point - b_sphere_center).length
        for point in points_co_global
    ) if b_sphere_center else None
    return b_sphere_center, b_sphere_radius


def normalize_scene(scale=None, offset=None, use_bounding_sphere=False, target_scale=0.5) -> tuple[float, mathutils.Vector]:
    """
    Normalize the scene by scaling and translating all objects.

    :param scale: scale factor
    :param offset: translation offset
    :param use_bounding_sphere: if True, use the bounding sphere to compute the scale factor
    :param target_scale: the target scale factor
    :return: scale factor and translation offset
    """

    if scale is None:
        if not use_bounding_sphere:
            bbox_min, bbox_max = scene_bbox()
            scale = target_scale / max(bbox_max - bbox_min)
        else:
            center, radius = scene_sphere()
            scale = target_scale / radius
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    if offset is None:
        bbox_min, bbox_max = scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return scale, offset


def scale_from_trimesh(filename, use_bounding_sphere=False) -> float:
    """
    Compute the scale factor from a trimesh file. Some GLB models contains animation data, which can effect the scale
    factor. This function uses trimesh to compute the scale factor, which can avoid the animation data.

    :param filename: the file name of the 3D object
    :param use_bounding_sphere: if True, use the bounding sphere to compute the scale factor
    """

    import trimesh
    mesh = trimesh.load(filename, force="mesh")
    if use_bounding_sphere:
        radius = mesh.bounding_sphere.primitive.radius
        scale = 0.5 / radius
    else:
        bbox_min = mesh.bounds[0]
        bbox_max = mesh.bounds[1]
        scale = 1.0 / max(bbox_max - bbox_min)
    return scale
