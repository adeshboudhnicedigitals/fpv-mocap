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
  to rest), but verify by eye.

## Camera and scale for the retarget

- The glb files contain no camera. Put a camera object in the scene: at x = 0, centred between the shoulders,
  0.15 m in front of them and 0.22 m above them (the plan's FPP offsets): about (0, 0.173, 0.45) m after the 0.01
  scale, rotation euler (90 deg, 0, 180 deg) so it looks along -Y with +Z up. `cam_world` is that object's
  `matrix_world` (Blender camera convention, looks along local -Z), as `cam_to_world` expects.
- Arm reach with 0.01 scale is about 0.52 m shoulder to wrist, shorter than a human arm; if the left wrist target
  (support hand on the handguard) is out of reach, scale the arms armature up uniformly by one factor (try
  1.15-1.25) and keep the shoulder positions relative to the camera.
