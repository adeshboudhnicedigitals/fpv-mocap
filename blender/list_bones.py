"""blender -b -P blender/list_bones.py -- <file.fbx>: print objects, armatures and bone names."""

import sys

import bpy

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=path)
for obj in bpy.data.objects:
    dims = tuple(round(d, 3) for d in obj.dimensions)
    print(f"OBJECT {obj.name} type={obj.type} dims={dims} parent={obj.parent.name if obj.parent else None}")
    if obj.type == "ARMATURE":
        for bone in obj.data.bones:
            print(f"BONE {bone.name} parent={bone.parent.name if bone.parent else None}")
