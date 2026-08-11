import os
from typing import Union

import bpy
import numpy as np
from mathutils import Euler, Vector

from bpy_helper.io import get_the_one_node_with_type


def remove_all_lights() -> None:
    """
    Remove all lights in the scene
    """

    # remove all lights
    light_objs = []
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            # bpy.data.objects.remove(obj)
            light_objs.append(obj)
    for obj in light_objs:
        bpy.data.objects.remove(obj)

    world = bpy.context.scene.world
    nodes = world.node_tree.nodes

    # clear the world background links
    world.use_nodes = True
    if len(world.node_tree.nodes['Background'].inputs[0].links) > 0:
        world.node_tree.links.remove(world.node_tree.nodes['Background'].inputs[0].links[0])
    # clear the world background nodes
    for node in nodes:
        if node.type in ['TEX_ENVIRONMENT', 'TEX_COORD', 'MAPPING']:
            nodes.remove(node)


def create_point_light(location, power=800., rgb=(1., 1., 1.), hard_shadow=False, keep_other_lights=False) \
        -> bpy.types.Object:
    """
    Creates a point light at the given location.

    :param location: The location of the light
    :param power: The power of the light, default is 800
    :param rgb: The color of the light, default is white
    :param hard_shadow: If true, the shadow will be hard. Default is False
    :param keep_other_lights: If true, the other lights will not be removed. Default is False
    :return: The light object
    """

    if not keep_other_lights:
        remove_all_lights()

    # Set the background ambient light
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

    light_data = bpy.data.lights.new(name="PointLight", type="POINT")
    light_data.energy = power
    light_data.color = rgb
    if hard_shadow:
        light_data.shadow_soft_size = 0
    light_obj = bpy.data.objects.new("PointLight", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = location
    return light_obj


def create_area_light(location, power=800., size=5., keep_other_lights=False, color = (1.,1.,1.)) -> bpy.types.Object:
    """
    Creates an area light at the given location.

    :param location: The location of the light
    :param power: The power of the light, default is 800
    :param size: The size of the light, default is 5
    :param keep_other_lights: If true, the other lights will not be removed. Default is False
    :return: The light object
    """

    if not keep_other_lights:
        remove_all_lights()

    light_data = bpy.data.lights.new(name="AreaLight", type="AREA")
    light_data.size = size
    light_data.energy = power
    light_data.color = color
    light_data.shape = 'DISK'

    light_obj = bpy.data.objects.new("AreaLight", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = location

    direction = Vector([0, 0, 0]) - Vector(location)
    light_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    return light_obj


def create_directional_light(location, power=800., rgb=(1., 1., 1.), keep_other_lights=False) -> bpy.types.Object:
    """
    Creates a directional light at the given location.

    :param location: The location of the light
    :param power: The power of the light, default is 800
    :param rgb: The color of the light, default is white
    :param keep_other_lights: If true, the other lights will not be removed. Default is False
    :return: The light object
    """

    if not keep_other_lights:
        remove_all_lights()

    light_data = bpy.data.lights.new(name="DirectionalLight", type="SUN")
    light_data.energy = power
    light_data.color = rgb

    light_obj = bpy.data.objects.new("DirectionalLight", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = location

    direction = Vector([0, 0, 0]) - Vector(location)
    light_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    return light_obj


def set_env_light(path_to_hdr_file: str, strength: float = 1.0,
                  rotation_euler: Union[list, Euler, np.ndarray] = None, keep_other_lights: bool = False) -> None:
    """
    Sets the world background to the given hdr_file.

    :param path_to_hdr_file: Path to the .exr file
    :param strength: The brightness of the background.
    :param rotation_euler: The euler angles of the background.
    :param keep_other_lights: If true, the other lights will not be removed.
    """

    if not keep_other_lights:
        remove_all_lights()

    scene = bpy.context.scene
    world = scene.world
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    if rotation_euler is None:
        rotation_euler = [0.0, 0.0, 0.0]

    if not os.path.exists(path_to_hdr_file):
        raise FileNotFoundError(f"The given path does not exists: {path_to_hdr_file}")

    # add a texture node and load the image and link it
    texture_node = nodes.new(type="ShaderNodeTexEnvironment")
    texture_node.image = bpy.data.images.load(path_to_hdr_file, check_existing=True)

    # get the one background node of the world shader
    background_node = get_the_one_node_with_type(nodes, "Background")

    # link the new texture node to the background
    links.new(texture_node.outputs["Color"], background_node.inputs["Color"])

    # Set the brightness of the background
    background_node.inputs["Strength"].default_value = strength

    # add a mapping node and a texture coordinate node
    mapping_node = nodes.new("ShaderNodeMapping")
    tex_coords_node = nodes.new("ShaderNodeTexCoord")

    # link the texture coordinate node to mapping node
    links.new(tex_coords_node.outputs["Generated"], mapping_node.inputs["Vector"])

    # link the mapping node to the texture node
    links.new(mapping_node.outputs["Vector"], texture_node.inputs["Vector"])

    mapping_node.inputs["Rotation"].default_value = rotation_euler
