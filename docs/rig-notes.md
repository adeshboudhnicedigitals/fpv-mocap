# M4 rig notes (Lone Asset `M4.fbx`)

Inspected with Blender 5.2.3 LTS (`C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`) via `blender/list_bones.py`.
Source: `lone-asset/content/weapons/m4/source/M4.fbx`, copied to `rigs/M4.fbx` (gitignored).

## Scale
FBX imports in **metres** (gun `Main` longest side 0.865). `global_scale=1.0` is correct; no cm fix needed.

## Objects
- Armature object: `Armature`
- **Gun mesh: `Main`** (dims 0.067 x 0.238 x 0.865 m). Child mesh `Safety`; other parts: `Bolt`, `BoltCatch`, `Mag1`, `Mag2`, `Trigger`, `Bullet5_56x45mm`(.001)
- Hand mesh: `Hand_Mesh`; camera object: `FPS_Camera`; empty: `Hand`
- Export: `rigs/m4_gun.obj`, `DIMS_M (0.067, 0.238, 0.865)`. Exported in world space (FBX placement kept, verts around z~1.38), not recentred; OBJ bbox x 0.067, y 0.865 (long axis, y -0.632..0.233), z 0.238 (1.30..1.54).

## Bones (armature `Armature`)
- Root `M4_Root`; gun bone **`Main_j`** (child of M4_Root; children `Mag1_j`, `Mag2_j`, `Bolt_j`, `BoltCatch_j`, `Trigger_j`)
- Camera bone: **`FPS_Camera_j`** (child of `M4_Root`); `Spine` is also a child of `M4_Root`
- Arms: `Spine` -> `RightArm`/`LeftArm` -> `RightForeArm`/`LeftForeArm` -> `RightHand`/`LeftHand`
- Fingers, 4 bones each (note `4` exists, tip): `{Right,Left}Hand{Thumb,Index,Middle,Ring,Pinky}{1,2,3,4}`, chained 1->2->3->4, 1 parented to the hand. E.g. `RightHandIndex1..4`.
- MANO finger order to use: index, middle, pinky, ring, thumb (names `Index`, `Middle`, `Pinky`, `Ring`, `Thumb`).

## Notes for retargeting
- Only `Main` is exported as the tracking mesh, on purpose: the gun is tracked as one rigid body, so moving parts (`Bolt`, `Mag1`, `Trigger`, ...) are left out.
- The OBJ export uses `forward_axis="Y"`, `up_axis="Z"`: that is Blender's identity axis conversion, so the OBJ frame equals the FBX world frame. FoundationPose poses therefore map FBX-world points to camera space, and the gun bone needs its rest world matrix `B_rest` applied (`GUN_REST` in `blender/retarget.py`, gun bone only; wrists use no rest matrix).
- MANO has 3 joints per finger, so map the rig's finger bones 1..3 (bone 4 is the tip and gets no rotation).
- The FBX may carry its own animation (a binary grep finds `AnimationStack`): `retarget.py` calls `arm.animation_data_clear()` and keys a fresh action named after the clip. `GUN_REST` and `cam_world` come from rest data (`arm.data.bones[...].matrix_local`), not the posed bone. Prefer the `FPS_Camera` object's `matrix_world` (it is a camera object) over the `FPS_Camera_j` bone.
- Pending check (Task 9 Step 2): `export_gun_mesh.py` exported the mesh at the import's current pose. Print the imported rig's gun bone pose matrix and confirm it equals its rest matrix at the frame used, or set `arm.data.pose_position = 'REST'` in both `blender/export_gun_mesh.py` and `blender/retarget.py` and re-export. `export_gun_mesh.py` has not been edited for this yet.
