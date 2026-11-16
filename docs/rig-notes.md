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
