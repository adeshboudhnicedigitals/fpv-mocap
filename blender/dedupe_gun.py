"""blender -b -P blender/dedupe_gun.py -- <in.glb> <out.glb> <name of the duplicate root to delete>

Deletes one node and everything under it, then exports a .glb. Used on m4a1.glb, which holds two
identical rifles (`base` and `base.001`) at the same place.
"""

import sys

import bpy

src, dst, name = sys.argv[sys.argv.index("--") + 1 :][:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
root = bpy.data.objects[name]
doomed = [root, *root.children_recursive]
before = len(bpy.data.objects)
for obj in doomed:
    bpy.data.objects.remove(obj, do_unlink=True)
print(f"DEDUPE removed {len(doomed)} objects, {before} -> {len(bpy.data.objects)}")
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
print("DEDUPE wrote", dst)
