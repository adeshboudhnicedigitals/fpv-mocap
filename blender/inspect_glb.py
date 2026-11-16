"""blender -b -P blender/inspect_glb.py -- <file.glb>: print armature scale and key bone positions, or object dims."""

import sys

import bpy

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
for obj in bpy.data.objects:
    if obj.type == "ARMATURE":
        print(f"ARMATURE {obj.name} scale={tuple(round(v, 4) for v in obj.scale)} "
              f"loc={tuple(round(v, 3) for v in obj.location)} rot={tuple(round(v, 3) for v in obj.rotation_euler)}")
        for b in obj.data.bones:
            if b.name.startswith(("R_arm", "R_elbow", "R_wrist", "R_point", "L_arm", "_root")):
                h = obj.matrix_world @ b.head_local
                t = obj.matrix_world @ b.tail_local
                print(f"BONE {b.name} parent={b.parent.name if b.parent else None} "
                      f"head={tuple(round(v, 3) for v in h)} tail={tuple(round(v, 3) for v in t)} "
                      f"len={round((t - h).length, 3)}")
    elif obj.type == "MESH":
        print(f"MESH {obj.name} dims={tuple(round(d, 3) for d in obj.dimensions)} parent={obj.parent.name if obj.parent else None}")
