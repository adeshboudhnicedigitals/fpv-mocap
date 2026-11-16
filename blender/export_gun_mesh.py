"""blender -b -P blender/export_gun_mesh.py -- <gun.glb> <root node name> <out.obj>

Exports the meshes under one root node as a single OBJ in metres, in Blender world axes (Z up),
for FoundationPose. Loose rounds and shells are skipped: they are not part of the rigid rifle shape.
"""

import sys

import bpy

SKIP_PREFIXES = ("shell", "bullet")

src, root_name, out = sys.argv[sys.argv.index("--") + 1 :][:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
root = bpy.data.objects[root_name]
meshes = [o for o in root.children_recursive if o.type == "MESH" and not o.name.lower().startswith(SKIP_PREFIXES)]
bpy.ops.object.select_all(action="DESELECT")
for obj in meshes:
    obj.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.wm.obj_export(
    filepath=out, export_selected_objects=True, export_materials=False, apply_modifiers=True,
    global_scale=1.0, forward_axis="Y", up_axis="Z",
)
from mathutils import Vector

corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = [min(c[i] for c in corners) for i in range(3)]
hi = [max(c[i] for c in corners) for i in range(3)]
print("GUNMESH parts", len(meshes), "bbox_min", [round(v, 3) for v in lo], "bbox_max", [round(v, 3) for v in hi],
      "dims", [round(h - l, 3) for l, h in zip(lo, hi)])
