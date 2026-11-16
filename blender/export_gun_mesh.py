"""blender -b -P blender/export_gun_mesh.py -- <rig.fbx> <mesh object name> <out.obj>

Exports one mesh in metres, with the object's own origin and axes, for FoundationPose.
"""

import sys

import bpy

rig, name, out = sys.argv[sys.argv.index("--") + 1 :][:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=rig)
obj = bpy.data.objects[name]
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.wm.obj_export(
    filepath=out, export_selected_objects=True, export_materials=False, apply_modifiers=True,
    global_scale=1.0, forward_axis="Y", up_axis="Z",
)
print("DIMS_M", tuple(round(d, 3) for d in obj.dimensions))
