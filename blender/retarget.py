"""blender -b -P blender/retarget.py -- <clip dir> <fps_arms.glb> <m4a1_single.glb>

Reads <clip dir>/aligned.npz and camera.json (poses in OpenCV camera space, metres), drives the fps_arms rig
and the m4a1 rifle, bakes one action named after the clip, and writes into <clip dir>:
  retarget_setup.blend  before baking: wrist empties, IK and Copy Rotation constraints live (for hand fixes)
  retarget.blend        after baking: plain keyed bones, constraints cleared
  <clip>.glb            arms (mesh + armature + baked action) and the rifle; no camera, no empties
Asset facts (units, axes, bone names, camera placement) are in docs/rig-notes.md.
"""

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mocap.transforms import cam_to_world, finger_local_rotation  # noqa: E402

# --- tunables ------------------------------------------------------------------------------------------------
ARM_SCALE = 0.01  # fps_arms.glb is in centimetres
ARMS_EXTRA_SCALE = 1.0  # uniform reach factor (try 1.15-1.25 if the support hand cannot reach the handguard)
CAM_AHEAD_OF_SHOULDERS = 0.15  # metres, along the arms' forward (-Y)
CAM_ABOVE_SHOULDERS = 0.22  # metres, along +Z
CAM_EULER_DEG = (90.0, 0.0, 180.0)  # camera looks along -Y with +Z up
GUN_ROOT_NODE = "base"  # the rifle subtree of m4a1_single.glb; everything else from that file is deleted
# MANO joint -> bone local rotation is C^-1 @ R @ C (see mocap.transforms.finger_local_rotation). Identity
# assumes the MANO joint axes equal the bone's rest axes; tune by eye (plan Task 9 Step 4).
FINGER_BASIS = np.eye(3)
# WiLoR estimates left hands on a flipped crop. True un-mirrors left finger rotations (reflect x) first.
MIRROR_LEFT = False
# --- rig names (prefixes, because the numeric suffixes are irregular) ---------------------------------------
SIDES = {"left": "L", "right": "R"}
MANO_FINGERS = ("point", "middle", "pink", "ring", "thumb")  # MANO order: index, middle, pinky, ring, thumb


def bone_name(arm, prefix: str) -> str:
    matches = [b.name for b in arm.data.bones if b.name.startswith(prefix)]
    if len(matches) != 1:
        raise SystemExit(f"retarget: expected one bone starting with {prefix!r} in {arm.name}, found {matches}")
    return matches[0]


def import_glb(path: Path) -> list:
    if not path.is_file():
        raise SystemExit(f"retarget: missing {path}")
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [o for o in bpy.data.objects if o not in before]


def to_matrix(T: np.ndarray) -> Matrix:
    return Matrix(np.asarray(T, dtype=float).tolist())


def key_object(obj, T_world: Matrix, frame: int, prev_quat):
    """Keys location + quaternion of an object at a world matrix; keeps the quaternion sign continuous."""
    obj.matrix_world = T_world
    if prev_quat is not None:
        obj.rotation_quaternion.make_compatible(prev_quat)
    obj.keyframe_insert("location", frame=frame)
    obj.keyframe_insert("rotation_quaternion", frame=frame)
    return obj.rotation_quaternion.copy()


def mesh_world_coords(obj) -> np.ndarray:
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    co = np.array([obj.matrix_world @ v.co for v in me.vertices])
    ev.to_mesh_clear()
    return co


argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
if len(argv) < 3:
    raise SystemExit("usage: blender -b -P blender/retarget.py -- <clip dir> <fps_arms.glb> <m4a1_single.glb>")
clip, arms_path, gun_path = (Path(a).resolve() for a in argv[:3])
for need in ("aligned.npz", "camera.json"):
    if not (clip / need).is_file():
        raise SystemExit(f"retarget: missing {clip / need}")
aligned = np.load(clip / "aligned.npz")
camera = json.loads((clip / "camera.json").read_text())
N = len(aligned["gun_valid"])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
scene = bpy.context.scene
scene.name = clip.name
scene.render.fps = round(camera["fps"])
scene.frame_start, scene.frame_end = 1, N

# --- arms: unparent from the Sketchfab empties, delete them, scale to metres and apply ----------------------
arms_objs = import_glb(arms_path)
arm = next(o for o in arms_objs if o.type == "ARMATURE")
arms_meshes = [o for o in arms_objs if o.type == "MESH" and any(m.type == "ARMATURE" and m.object == arm
                                                                  for m in o.modifiers)]
for pb in arm.pose.bones:
    pb.custom_shape = None  # the importer's Icosphere bone shape, deleted below
world = arm.matrix_world.copy()
arm.parent = None
arm.matrix_world = world
for obj in arms_objs:
    if obj is not arm and obj not in arms_meshes:
        bpy.data.objects.remove(obj, do_unlink=True)
for obj in (arm, *arms_meshes):
    obj.animation_data_clear()
arm.scale = [s * ARM_SCALE * ARMS_EXTRA_SCALE for s in arm.scale]
bpy.ops.object.select_all(action="DESELECT")
for obj in (arm, *arms_meshes):
    obj.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
for pb in arm.pose.bones:
    pb.matrix_basis = Matrix()
    pb.rotation_mode = "QUATERNION"
bpy.context.view_layer.update()

B = {}  # side -> {"arm", "elbow", "wrist", "fingers": [15 names in MANO order, bones 1..3]}
for side, S in SIDES.items():
    B[side] = {part: bone_name(arm, f"{S}_{part}_") for part in ("arm", "elbow", "wrist")}
    B[side]["fingers"] = [bone_name(arm, f"{S}_{f}{k}_") for f in MANO_FINGERS for k in (1, 2, 3)]

# --- bone tails: IK drives the TAIL of the constrained bone, and the imported tails are guesses ----------------
rest_before = np.concatenate([mesh_world_coords(o) for o in arms_meshes])
bpy.ops.object.mode_set(mode="EDIT")
eb = arm.data.edit_bones
for side in SIDES:
    eb[B[side]["arm"]].tail = eb[B[side]["elbow"]].head.copy()
    eb[B[side]["elbow"]].tail = eb[B[side]["wrist"]].head.copy()
bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.view_layer.update()
rest_shift = np.abs(np.concatenate([mesh_world_coords(o) for o in arms_meshes]) - rest_before).max()
print(f"RETARGET tail fix: max rest vertex shift {rest_shift:.2e} m")
if rest_shift > 1e-5:
    raise SystemExit(f"retarget: fixing the arm bone tails moved the rest mesh by {rest_shift} m")

# --- gun: keep the `base` subtree, parent it to weapon_root at the world origin (GUN_REST = identity) -------
gun_objs = import_glb(gun_path)
base = next((o for o in gun_objs if o.name == GUN_ROOT_NODE), None)
if base is None:
    raise SystemExit(f"retarget: no node {GUN_ROOT_NODE!r} in {gun_path}")
rifle = {base, *base.children_recursive}
weapon_root = bpy.data.objects.new("weapon_root", None)
scene.collection.objects.link(weapon_root)
weapon_root.rotation_mode = "QUATERNION"
base_world = base.matrix_world.copy()
base.parent = weapon_root
base.matrix_world = base_world
for obj in gun_objs:
    if obj not in rifle:
        bpy.data.objects.remove(obj, do_unlink=True)
for obj in rifle:
    obj.animation_data_clear()
for action in list(bpy.data.actions):  # neither glb should carry animation; drop anything imported
    bpy.data.actions.remove(action)

# --- camera: centred between the shoulders, ahead of and above them --------------------------------------
shoulders = [arm.matrix_world @ arm.data.bones[B[side]["arm"]].head_local for side in SIDES]
cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(cam)
cam.location = (shoulders[0] + shoulders[1]) / 2 + Vector((0, -CAM_AHEAD_OF_SHOULDERS, CAM_ABOVE_SHOULDERS))
cam.rotation_euler = [math.radians(a) for a in CAM_EULER_DEG]
K = np.array(camera["K"])
W, H = camera["width"], camera["height"]
cam.data.sensor_fit = "HORIZONTAL"
cam.data.lens = K[0, 0] * cam.data.sensor_width / W  # same horizontal field of view as the footage
cam.data.shift_x = (W / 2 - K[0, 2]) / W
cam.data.shift_y = (K[1, 2] - H / 2) / W
scene.camera = cam
scene.render.resolution_x, scene.render.resolution_y = W, H
bpy.context.view_layer.update()
cam_world = np.array(cam.matrix_world)

# --- wrist empties and constraints --------------------------------------------------------------------------
targets = {}
for side in SIDES:
    empty = bpy.data.objects.new(f"ik_target_{side}", None)
    scene.collection.objects.link(empty)
    empty.empty_display_type, empty.empty_display_size = "ARROWS", 0.05
    empty.rotation_mode = "QUATERNION"
    targets[side] = empty
    ik = arm.pose.bones[B[side]["elbow"]].constraints.new("IK")
    ik.target, ik.chain_count = empty, 2
    copy_rot = arm.pose.bones[B[side]["wrist"]].constraints.new("COPY_ROTATION")
    copy_rot.target = empty

# --- keys: one action named after the clip carries the arms and weapon_root (one glTF animation) ----------
action = bpy.data.actions.new(clip.name)
for obj in (arm, weapon_root):
    obj.animation_data_create().action = action
finger_basis = {"right": FINGER_BASIS, "left": FINGER_BASIS}
prev = {}  # last keyed quaternion per object/bone, for sign continuity
last_gun = None
for i in range(N):
    frame = i + 1
    if aligned["gun_valid"][i]:
        last_gun = to_matrix(cam_to_world(cam_world, aligned["T_cam_gun"][i]))
    if last_gun is not None:  # invalid frames hold the last valid pose (leading ones hold the first key)
        prev["gun"] = key_object(weapon_root, last_gun, frame, prev.get("gun"))
    for side, empty in targets.items():
        if not aligned[f"{side}_valid"][i]:
            continue
        T = to_matrix(cam_to_world(cam_world, aligned[f"{side}_T_cam_wrist"][i]))
        prev[side] = key_object(empty, T, frame, prev.get(side))
        local = finger_local_rotation(aligned[f"{side}_finger_rot"][i], finger_basis[side],
                                      mirror=MIRROR_LEFT and side == "left")
        for name, R in zip(B[side]["fingers"], local):
            pb = arm.pose.bones[name]
            q = to_matrix(R).to_quaternion()
            if name in prev:
                q.make_compatible(prev[name])
            pb.rotation_quaternion = prev[name] = q
            pb.keyframe_insert("rotation_quaternion", frame=frame)
if last_gun is None:
    raise SystemExit("retarget: no frame has gun_valid = True")

scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(clip / "retarget_setup.blend"), copy=True)
print("RETARGET wrote", clip / "retarget_setup.blend")

# --- bake the arms with visual keying, constraints cleared ------------------------------------------------
bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
bpy.ops.nla.bake(frame_start=1, frame_end=N, only_selected=False, visual_keying=True, clear_constraints=True,
                 use_current_action=True, bake_types={"POSE"})
bpy.ops.object.mode_set(mode="OBJECT")
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(clip / "retarget.blend"))
print("RETARGET wrote", clip / "retarget.blend")

# --- export: arms + rifle, no camera, no empties ----------------------------------------------------------
bpy.ops.object.select_all(action="DESELECT")
for obj in (arm, *arms_meshes, weapon_root, *rifle):
    obj.select_set(True)
glb = clip / f"{clip.name}.glb"
bpy.ops.export_scene.gltf(filepath=str(glb), export_format="GLB", use_selection=True, export_animations=True,
                          export_current_frame=False)
print("RETARGET exported", glb)
