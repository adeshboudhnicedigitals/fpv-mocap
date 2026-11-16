# GPU stages for a clip: hands (WiLoR), depth (Depth Anything V2), gun mask (SAM 2), gun pose (FoundationPose).
# Percent format: each `# %%` is a notebook cell. Run on Kaggle/Colab; needs CUDA. Do not import on the laptop.
# Every line that depends on an external repo's API is marked ADAPTER and prints keys/shapes on the first
# run so they can be confirmed. Everything else only uses this repo's contracts.

# %% setup
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/kaggle/working/fpv-mocap")  # Colab: Path("/content/fpv-mocap")
CLIP = ROOT / "clips" / "m4_pullout"
sys.path.insert(0, str(ROOT))
from mocap.contracts import save_gun, save_hands  # noqa: E402
from mocap.frames import list_frames  # noqa: E402

meta = json.loads((CLIP / "meta.json").read_text())
cam = json.loads((CLIP / "camera.json").read_text())
K = np.array(cam["K"])
frames = list_frames(CLIP / "rgb")
N = len(frames)
init = meta["init_frame"] - 1  # 0-based index

# %% install (once per session)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wilor-mini", "sam2", "trimesh"], check=True)
# Depth Anything V2 and FoundationPose are cloned and built as their READMEs describe:
#   git clone https://github.com/DepthAnything/Depth-Anything-V2 ; cd metric_depth ; download the
#   depth_anything_v2_metric_hypersim_vitl.pth checkpoint
#   git clone https://github.com/NVlabs/FoundationPose ; build_all.sh (compiles CUDA extensions;
#   save the build as a Kaggle Dataset so later sessions skip it)

# %% hands: WiLoR (ADAPTER: confirm output keys on the first run)
import torch  # noqa: E402
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (  # noqa: E402  ADAPTER
    WiLorHandPose3dEstimationPipeline,
)

pipe = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float16)  # ADAPTER: constructor args

# ADAPTER switches, decide after reading the first-run printout:
USE_KEYPOINT0_AS_WRIST = True  # wrist translation = keypoint 0 (+ cam_t) rather than pred_cam_t_full
LEFT_NEEDS_UNMIRROR = False  # set True if left-hand rotations/translations come out in the mirrored frame
MIRROR_X = np.diag([-1.0, 1.0, 1.0])


def to_rotmat(v, n):
    """ADAPTER: WiLoR may return rotation matrices or axis-angle. Returns (n, 3, 3) float64 either way."""
    a = np.asarray(v, dtype=np.float64)
    if a.size == n * 9:
        return a.reshape(n, 3, 3)
    if a.size == n * 3:
        return Rotation.from_rotvec(a.reshape(n, 3)).as_matrix()
    raise ValueError(f"unexpected rotation shape {a.shape} for {n} joints")


def empty_hand():
    return {
        "T_cam_wrist": np.tile(np.eye(4), (N, 1, 1)),
        "finger_rot": np.tile(np.eye(3), (N, 15, 1, 1)),
        "keypoints": np.zeros((N, 21, 3)),
        "valid": np.zeros(N, bool),
        "score": np.zeros(N),
    }


hands = {"left": empty_hand(), "right": empty_hand()}
for i, path in enumerate(frames):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    for out in pipe.predict(rgb):  # ADAPTER: returns a list of dicts, one per detected hand
        p = out["wilor_preds"]  # ADAPTER
        if i == init:
            print("WiLoR keys:", list(out.keys()), list(p.keys()))  # ADAPTER check
            print({k: np.asarray(v).shape for k, v in p.items()})  # ADAPTER check: axis-angle vs matrix shapes
        side = "right" if out["is_right"] else "left"  # ADAPTER
        h = hands[side]
        R = to_rotmat(p["global_orient"], 1)[0]  # ADAPTER
        cam_t = np.asarray(p["pred_cam_t_full"], dtype=np.float64).reshape(3)  # ADAPTER
        kp = np.asarray(p["pred_keypoints_3d"], dtype=np.float64).reshape(21, 3)  # ADAPTER
        fingers = to_rotmat(p["hand_pose"], 15)  # ADAPTER
        if side == "left" and LEFT_NEEDS_UNMIRROR:
            R = MIRROR_X @ R @ MIRROR_X
            fingers = MIRROR_X @ fingers @ MIRROR_X
            kp = kp * [-1.0, 1.0, 1.0]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = kp[0] + cam_t if USE_KEYPOINT0_AS_WRIST else cam_t
        h["T_cam_wrist"][i] = T
        h["finger_rot"][i] = fingers
        h["keypoints"][i] = kp + cam_t
        h["valid"][i] = True
        h["score"][i] = 1.0  # ADAPTER: no detection confidence assumed
# Frames where a side was not detected keep their identity poses and valid=False (empty_hand defaults).
save_hands(CLIP / "hands.npz", hands)
print({s: int(hands[s]["valid"].sum()) for s in hands}, "valid frames of", N)

# %% depth: Depth Anything V2 metric (run from the cloned metric_depth folder; ADAPTER)
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402  ADAPTER

model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], max_depth=20)
model.load_state_dict(torch.load("checkpoints/depth_anything_v2_metric_hypersim_vitl.pth", map_location="cpu"))
model = model.to("cuda").eval()
(CLIP / "depth").mkdir(exist_ok=True)
for i, path in enumerate(frames):
    d = model.infer_image(cv2.imread(str(path))).astype(np.float32)  # ADAPTER: (H, W) metres expected
    if i == init:
        print("depth shape/range:", d.shape, float(d.min()), float(d.max()))  # ADAPTER check
    np.save(CLIP / "depth" / f"{i + 1:05d}.npy", d)

# %% gun mask on the init frame: SAM 2
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
init_rgb = cv2.cvtColor(cv2.imread(str(frames[init])), cv2.COLOR_BGR2RGB)
predictor.set_image(init_rgb)
masks, scores, _ = predictor.predict(
    point_coords=np.array([meta["gun_click"]]), point_labels=np.array([1]), multimask_output=False
)
(CLIP / "masks").mkdir(exist_ok=True)
cv2.imwrite(str(CLIP / "masks" / "gun_mask.png"), masks[0].astype(np.uint8) * 255)
overlay = init_rgb.copy()
overlay[masks[0] > 0] = (0.5 * overlay[masks[0] > 0] + [127, 0, 0]).astype(np.uint8)
cv2.imwrite(str(CLIP / "masks" / "gun_mask_check.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
# Look at gun_mask_check.png: the mask must cover the rifle only. If not, change gun_click in meta.json.

# %% gun: FoundationPose, forward then backward from the init frame (ADAPTER)
import trimesh  # noqa: E402
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, dr  # noqa: E402  ADAPTER

mesh = trimesh.load(ROOT / "rigs" / "m4_gun.obj", force="mesh")
est = FoundationPose(  # ADAPTER: constructor args
    model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
    scorer=ScorePredictor(), refiner=PoseRefinePredictor(), glctx=dr.RasterizeCudaContext(),
)
mask = cv2.imread(str(CLIP / "masks" / "gun_mask.png"), cv2.IMREAD_GRAYSCALE) > 0

T_gun = np.tile(np.eye(4), (N, 1, 1))  # invalid frames keep identity
gun_valid = np.zeros(N, bool)


def rgb_depth(i):
    return (cv2.cvtColor(cv2.imread(str(frames[i])), cv2.COLOR_BGR2RGB), np.load(CLIP / "depth" / f"{i + 1:05d}.npy"))


for order in (range(init, N), range(init, -1, -1)):
    rgb, depth = rgb_depth(init)
    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)  # ADAPTER
    print("register pose shape:", np.asarray(pose).shape)  # ADAPTER check: expect (4, 4)
    for i in order:
        if i != init:
            rgb, depth = rgb_depth(i)
            pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=2)  # ADAPTER
        T_gun[i] = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        gun_valid[i] = True
# Frames before the gun enters the view are not tracked reliably: after looking at overlay.mp4 set
# gun_valid[:first_visible] = False and T_gun[:first_visible] = np.eye(4), then save again.
save_gun(CLIP / "gun.npz", {"T_cam_gun": T_gun, "valid": gun_valid, "score": gun_valid.astype(float)})
