import bpy


def set_smooth_shading() -> None:
    """
    Sets smooth shading for all objects in the scene.
    """

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for m in meshes:
        for f in m.data.polygons:
            f.use_smooth = True
        m.data.update()
