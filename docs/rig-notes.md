# Assets for the retarget: `fps_arms.glb` and `m4a1_single.glb`

These replace the Lone Asset `M4.fbx` rig (the old notes are in git history, commit 30639fd). Inspected with
Blender 5.2.3 (`C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`) through the glTF importer, using
`blender/inspect_glb.py`. Both files are Sketchfab exports that sit in the repo root, uncommitted.

## Gun: `m4a1_single.glb`

- Made from `m4a1.glb` with `blender/dedupe_gun.py`: the original held two identical rifles (`base`, `base.001`).
- Rigid, **no skin, no animation**. Metres. Hierarchy: `Sketchfab_model/…fbx/RootNode/<parts>`; the rifle is the
  subtree under node **`base`** (21 meshes, 6 of them rounds and shells). Still present and NOT part of the
  rifle: a spare magazine `magazine.002` (8 meshes) and about 28 loose `shell*` casings.
- After the glTF import, in Blender world axes (Z up), the rifle subtree minus rounds and shells (15 parts)
  measures 0.069 x 0.882 x 0.371 m, lying along **+Y**, bbox y -0.466..0.416, z -0.021..0.349. Nodes carry
  transforms, so do not trust per-mesh dimensions.
- Checked in a side render: the **muzzle is at +Y, the stock at -Y**, sights up (+Z). The bore is tilted about
  11.8 deg nose-up: `base` is rotated 78.2 deg (not 90) about X, so its local +Z, the bore axis, is
  (0, 0.979, 0.204) in world. The origin sits near the bottom of the magazine well, about 0.2 m below the
  receiver top; the pistol grip is around (0, -0.15, 0.08). So a rifle held forward in the camera view is the
  rest mesh turned 180 deg about Z (and the 11.8 deg tilt removed); with real data FoundationPose supplies that.
- `blender/retarget.py` deletes everything from this file that is not under `base` (spare magazine, loose shells).
- `rigs/m4_gun.obj` (gitignored) is that rifle exported with `blender/export_gun_mesh.py`: world axes, metres, not
  recentred. FoundationPose returns poses of THIS frame (the Blender world frame of the import). The previous
  tracking mesh is kept as `rigs/m4_gun_lone_asset.obj`.
- Because the rifle has no bone, the animated gun is an empty `weapon_root` at the world origin with the imported
  `base` subtree parented to it, so the rest matrix of the gun is the identity and the gun pose is simply
  `cam_world @ FLIP @ T_cam_gun`.

## Arms: `fps_arms.glb`

- One skinned mesh, one skin of 47 joints, **no animation**. Armature object `Object_3`, object scale 1, but the
  numbers are in **centimetres**: upper arm 25.9, forearm about 25.5 (so apply 0.01 to get metres, about 0.26 m
  each). Also checked in Blender: an `Icosphere` object appears on import; it is the importer's bone-display
  helper, not part of the asset.
- Faces Blender **-Y** (glTF forward maps to -Y). Rest shoulders at (+-24.58, 32.32, 22.99) cm; rest wrists at
  about (+-19.5, -16.9, 10.6) cm; fingertips about y = -26 cm.
- Bones (parent chain): `_rootJoint` > `L_arm_01` > `L_elbow_02` > `L_wrist_03` > fingers; `_rootJoint` >
  `R_arm_023` > `R_elbow_024` > `R_wrist_025` > fingers.
  - `*_arm_*` is the upper arm, `*_elbow_*` the forearm (its head is the elbow), `*_wrist_*` the hand.
  - Fingers, 4 bones each, chained 1 > 2 > 3 > 4, bone 1 parented to the wrist bone: `thumb`, `point` (index),
    `middle`, `ring`, `pink` (little). Example: `R_point1_030`.
  - **The numeric suffixes are irregular** (`L_middle2_00`, `L_point1_08`, `L_point3_010`), so look bones up by
    the prefix `f"{S}_{finger}{k}_"` with S in `L`/`R`, never by full name.
  - MANO has 3 joints per finger: use bones 1..3, leave 4 (the tip) alone. MANO finger order is
    index, middle, pinky, ring, thumb = `point`, `middle`, `pink`, `ring`, `thumb`.
- **Bone tails are importer guesses.** `R_elbow_024` has head at the elbow but its tail points up the Z axis
  (length 25), not at the wrist. Blender's IK drives the TAIL of the constrained bone to the target, so before
  adding IK set each tail to its child's head in edit mode: `arm_01.tail = elbow_02.head`,
  `elbow_02.tail = wrist_03.head`. At identity pose this changes nothing on screen (deformation is relative
  to rest). Verified by `blender/retarget.py`: largest rest vertex shift after the fix is 7e-8 m.
- The finger bone tails are importer guesses too (their local Y points roughly up +Z, not along the finger).
  The rest pose is a loose grip (fingers curled). With `FINGER_BASIS` = identity, a 40 deg rotation about
  MANO/bone local X **un-curls** the index finger (seen in a render), so `FINGER_BASIS` must be tuned on real
  data; leave the finger tails alone, because the finger rotations are keyed relative to these rest axes.
- Import also brings the Sketchfab empties (`Sketchfab_model`, `<hash>.fbx`, `RootNode`), an empty `Object_5`
  under the armature, and the `Icosphere` that every pose bone uses as custom shape.
  `retarget.py` unparents `Object_3` keeping its world matrix (mesh `Object_6` stays its child), deletes the
  rest, then applies the scale.

## Camera and scale for the retarget

- The glb files contain no camera. Put a camera object in the scene: at x = 0, centred between the shoulders,
  0.15 m in front of them and 0.22 m above them (the plan's FPP offsets): about (0, 0.173, 0.45) m after the 0.01
  scale, rotation euler (90 deg, 0, 180 deg) so it looks along -Y with +Z up. `cam_world` is that object's
  `matrix_world` (Blender camera convention, looks along local -Z), as `cam_to_world` expects.
- Verified: the camera lands at (0, 0.1732, 0.4499), lens 18 mm = 90 deg horizontal FOV for fx 960 / 1920 px. At
  rest the arms are out of frame below (wrists about 45 deg under the view axis; vertical half-FOV is 29 deg).
- Arm reach with 0.01 scale is about 0.52 m shoulder to wrist, shorter than a human arm; if the left wrist target
  (support hand on the handguard) is out of reach, scale the arms armature up uniformly by one factor (try
  1.15-1.25) and keep the shoulder positions relative to the camera.
  Measured on the synthetic clip (gun 0.45 m ahead, right of centre; support hand on the handguard): the left
  target was 0.81 m from the left shoulder, 0.30 m out of reach at 1.0 and still 0.20 m at 1.25 (needs about
  1.6, which is not plausible). On real data move nothing until the tracked distances are known: if the support
  wrist is out of reach, the gun or camera placement is likely off, not the arm length.

## Blender 5.x notes (5.2.3, headless)

- `blender/retarget.py` ran first time with no API change from the 4.x-style calls; nothing needed a context
  override in `-b` mode. Details worth knowing:
- Slotted actions: one `bpy.data.actions.new(clip)` assigned to both the armature and `weapon_root`, then
  `keyframe_insert(..., frame=f)`, auto-creates two slots (`OBObject_3`, `OBweapon_root`). The glTF exporter
  writes that as ONE animation named after the action (143 channels), which is what Godot should play.
- F-curves live in channelbags: use `bpy_extras.anim_utils.action_get_channelbag_for_slot(action, slot)`.
- `bpy.ops.nla.bake(frame_start=1, frame_end=N, only_selected=False, visual_keying=True,
  clear_constraints=True, use_current_action=True, bake_types={"POSE"})` works with the armature active in
  POSE mode; it keys location, rotation and scale on every bone and removes all bone constraints.
- `object.keyframe_insert(path, frame=f)` and `pose_bone.keyframe_insert(path, frame=f)` work without
  `frame_set`, which keeps the keying loop fast.
- `transform_apply(scale=True)` with the armature and its skinned child mesh both selected scales the bones,
  the mesh and the deformation consistently (both end with scale 1).
- `matrix_world` is float32: round trips through Blender are exact to about 2e-5, not 1e-8.
- Re-importing the exported glb into a factory scene retimes keys to 24 fps: set `scene.render.fps = 60`
  before `import_scene.gltf` when checking an export.
