# GPU stages for a clip: hands (WiLoR), depth (Depth Anything V2), gun mask (SAM 2), gun pose (FoundationPose).
# Percent format: each `# %%` is a notebook cell. Run on Kaggle/Colab; needs CUDA. Do not import on the laptop.
# Every line that depends on an external repo's API is marked ADAPTER and prints keys/shapes on the first
# run so they can be confirmed. Everything else only uses this repo's contracts.

# %% setup
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import importlib
import numpy as np
from scipy.spatial.transform import Rotation

BASE = Path("/content") if Path("/content").exists() else Path("/kaggle/working")  # Colab or Kaggle
ROOT = BASE / "fpv-mocap"
CLIP = ROOT / "clips" / "m4_pullout"
DA_DIR = BASE / "Depth-Anything-V2" / "metric_depth"
FP_DIR = BASE / "FoundationPose"
# Set after viewing overlay.mp4: (first, last) 0-based inclusive frames where the gun track is good.
GUN_VALID_RANGE = (None, None)
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
# wilor-mini pins torch<=2.5, which would downgrade the runtime's torch, and needs chumpy, whose old setup.py
# fails under build isolation. So: wilor-mini without its dependencies, then the dependencies one by one.
# SAM 2 and WiLoR-mini are not on PyPI under these names: both install from git.
def pip(*args, check=True, **kw):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=check, **kw)


pip("--no-deps", "git+https://github.com/warmshao/WiLoR-mini")
pip("smplx==0.1.28", "transformers", "timm", "einops", "ultralytics==8.1.34", "huggingface_hub", "scikit-image", "roma", "trimesh")
pip("setuptools", "wheel")
pip("--no-build-isolation", "git+https://github.com/mattloper/chumpy")
pip("git+https://github.com/facebookresearch/sam2.git", env={**os.environ, "SAM2_BUILD_ALLOW_ERRORS": "1"})
# ultralytics 8.1.34 loads its weights with an old torch.load; torch>=2.6 would refuse them by default.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
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

printed_keys = False
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


# WiLoR's own YOLO detector misses rendered gloved hands, so boxes come from OWLv2 (zero-shot, text query)
# and go to WiLoR through predict_with_bboxes. Side is a heuristic: one box = left (the support hand,
# the right hand is behind the gun), two boxes = leftmost is left.
from transformers import Owlv2ForObjectDetection, Owlv2Processor  # noqa: E402
from torchvision.ops import nms  # noqa: E402

owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to("cuda").eval()
HAND_QUERY = "a hand wearing a black glove"
HAND_SCORE = 0.1  # lower if hands are missed, raise if the debug boxes land on sleeves or the gun


def find_hands(rgb):
    """Returns (boxes (n,4) xyxy float, scores (n,)), at most 2, sorted left to right."""
    h, w = rgb.shape[:2]
    inputs = owl_proc(text=[[HAND_QUERY]], images=rgb, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = owl(**inputs)
    side = max(h, w)  # OWLv2 pads the image to a square
    res = owl_proc.image_processor.post_process_object_detection(out, threshold=HAND_SCORE, target_sizes=torch.tensor([[side, side]]))[0]
    boxes, scores = res["boxes"].cpu(), res["scores"].cpu()
    keep = nms(boxes, scores, 0.3)[:2]
    boxes, scores = boxes[keep].numpy(), scores[keep].numpy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    order = np.argsort(boxes[:, 0]) if len(boxes) else []
    return boxes[order].astype(np.float64), scores[order]


(CLIP / "hand_check").mkdir(exist_ok=True)
hands = {"left": empty_hand(), "right": empty_hand()}
for i, path in enumerate(frames):
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes, box_scores = find_hands(rgb)
    if i % 10 == 0:  # look at these to check the boxes are on hands
        dbg = bgr.copy()
        for b in boxes:
            cv2.rectangle(dbg, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 4)
        cv2.imwrite(str(CLIP / "hand_check" / f"{i + 1:05d}.jpg"), cv2.resize(dbg, None, fx=0.5, fy=0.5))
    if len(boxes) == 0:
        continue
    is_rights = [0] if len(boxes) == 1 else [0, 1]
    for out, score in zip(pipe.predict_with_bboxes(rgb, boxes, is_rights), box_scores):
        p = out["wilor_preds"]  # ADAPTER
        if not printed_keys:
            printed_keys = True
            print("first detection at frame", i)
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
        h["score"][i] = float(score)
# Frames where a side was not detected keep their identity poses and valid=False (empty_hand defaults).
save_hands(CLIP / "hands.npz", hands)
print({s: int(hands[s]["valid"].sum()) for s in hands}, "valid frames of", N)
print("check boxes in", CLIP / "hand_check")

# %% depth: Depth Anything V2 metric (run from the cloned metric_depth folder; ADAPTER)
if not DA_DIR.exists():
    subprocess.run(["git", "clone", "https://github.com/DepthAnything/Depth-Anything-V2", str(DA_DIR.parent)], check=True)
ckpt = DA_DIR / "checkpoints" / "depth_anything_v2_metric_hypersim_vitl.pth"
if not ckpt.exists():
    from huggingface_hub import hf_hub_download
    hf_hub_download("depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
                    "depth_anything_v2_metric_hypersim_vitl.pth", local_dir=str(ckpt.parent))
sys.path.insert(0, str(DA_DIR))
importlib.invalidate_caches()  # DA_DIR was a missing path on the first try; Python cached that
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402  ADAPTER

model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], max_depth=20)
model.load_state_dict(torch.load(DA_DIR / "checkpoints" / "depth_anything_v2_metric_hypersim_vitl.pth", map_location="cpu"))
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

# %% FoundationPose install: clone, GPU extensions, native build, weights (ADAPTER: first run on Kaggle)
# The repo targets modern torch (its requirements.txt asks for numpy>=2 and a CUDA 12 torch), so the
# runtime's torch is kept. pytorch3d compiles from source: about 10-20 minutes the first time per session.
def sh(cmd, **kw):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True, **kw)


if not FP_DIR.exists():
    sh(f"git clone https://github.com/NVlabs/FoundationPose {FP_DIR}")
CUDA_HOME = os.environ.get("CUDA_HOME") or next((c for c in ("/usr/local/cuda", "/usr/local/cuda-12.4") if Path(c, "bin", "nvcc").exists()), "")
assert CUDA_HOME, "nvcc not found: FoundationPose needs the CUDA toolkit (pytorch3d and nvdiffrast compile against it)"
os.environ.update(CUDA_HOME=CUDA_HOME, PATH=f"{CUDA_HOME}/bin:{os.environ['PATH']}", FORCE_CUDA="1",
                  TORCH_CUDA_ARCH_LIST="7.5", PYOPENGL_PLATFORM="egl")  # 7.5 = T4; one arch compiles fastest
try:
    import pytorch3d  # noqa: F401
except ImportError:
    pip("--no-build-isolation", "git+https://github.com/facebookresearch/pytorch3d.git")
try:
    import nvdiffrast  # noqa: F401
except ImportError:
    pip("--no-build-isolation", "git+https://github.com/NVlabs/nvdiffrast.git")
pip("-r", str(FP_DIR / "requirements.txt"), "pybind11", "gdown")
if not list((FP_DIR / "mycpp" / "build").glob("mycpp*.so")):
    sh("apt-get install -y -q libboost-all-dev libeigen3-dev cmake libgl1 libegl1 > /dev/null")
    sh(f"cd {FP_DIR}/mycpp && rm -rf build && mkdir build && cd build && "
       f"cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$({sys.executable} -m pybind11 --cmakedir) "
       f"-DPYTHON_EXECUTABLE={sys.executable} && make -j$(nproc)")
for w in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):  # refiner, scorer
    if not (FP_DIR / "weights" / w).exists():
        import gdown
        gdown.download_folder("https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i",
                              output=str(FP_DIR / "weights"), quiet=True)
        break
assert (FP_DIR / "weights" / "2023-10-28-18-33-37").exists() and (FP_DIR / "weights" / "2024-01-11-20-02-45").exists(), \
    "weights missing: download both folders from the FoundationPose Google Drive into FoundationPose/weights/"

# %% gun: FoundationPose, register at init, track forward then backward (ADAPTER)
import gc  # noqa: E402

import trimesh  # noqa: E402

# FoundationPose needs most of a 15 GB T4 on its own. Run it on the second GPU when there is one (Kaggle
# T4 x2: the other stages stay on GPU 0); it builds its tensors on the current CUDA device.
# With a single GPU, free the earlier stages' models instead.
for name in ("pipe", "owl", "owl_proc", "model", "predictor"):
    globals().pop(name, None)
gc.collect()
torch.cuda.empty_cache()
FP_GPU = 1 if torch.cuda.device_count() > 1 else 0
torch.cuda.set_device(FP_GPU)
print(f"{torch.cuda.device_count()} GPUs; FoundationPose on cuda:{FP_GPU}, {torch.cuda.mem_get_info(FP_GPU)[0] / 2**30:.1f} GiB free")

sys.path.insert(0, str(FP_DIR))
sys.path.insert(0, str(FP_DIR / "mycpp" / "build"))
importlib.invalidate_caches()
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, dr  # noqa: E402  ADAPTER

mesh = trimesh.load(ROOT / "rigs" / "m4_gun.obj", force="mesh")
# FoundationPose scores ~250 pose hypotheses against the full-size depth map: at 1920x1080 that is ~2 GB per
# tensor and runs out of a 15 GB T4. Run it at FP_SCALE; K scales with the image, poses stay in the camera frame.
FP_SCALE = 0.5
K_fp = np.asarray(K, dtype=np.float64).copy()
K_fp[:2] *= FP_SCALE
mask = cv2.imread(str(CLIP / "masks" / "gun_mask.png"), cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, None, fx=FP_SCALE, fy=FP_SCALE, interpolation=cv2.INTER_NEAREST) > 0

T_gun = np.tile(np.eye(4), (N, 1, 1))  # invalid frames keep identity
gun_valid = np.zeros(N, bool)


def rgb_depth(i):
    rgb = cv2.cvtColor(cv2.imread(str(frames[i])), cv2.COLOR_BGR2RGB)
    depth = np.load(CLIP / "depth" / f"{i + 1:05d}.npy")
    return (cv2.resize(rgb, None, fx=FP_SCALE, fy=FP_SCALE, interpolation=cv2.INTER_AREA),
            cv2.resize(depth, None, fx=FP_SCALE, fy=FP_SCALE, interpolation=cv2.INTER_NEAREST))


# ADAPTER: FoundationPose loads assets by relative path, so it must run with cwd = its repo (from memory,
# confirm on the first run). All paths passed to it above/below are absolute.
prev_cwd = os.getcwd()
os.chdir(FP_DIR)
try:
    est = FoundationPose(  # ADAPTER: constructor args
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=ScorePredictor(), refiner=PoseRefinePredictor(), glctx=dr.RasterizeCudaContext(),
    )
    rgb, depth = rgb_depth(init)
    init_pose = np.asarray(est.register(K=K_fp, rgb=rgb, depth=depth, ob_mask=mask, iteration=5), dtype=np.float64)  # ADAPTER
    print("register pose shape:", init_pose.shape)  # ADAPTER check: expect (4, 4)
    init_pose = init_pose.reshape(4, 4)
    T_gun[init] = init_pose
    gun_valid[init] = True
    for order in (range(init + 1, N), range(init - 1, -1, -1)):
        # track_one follows the estimator's last pose, so the backward pass must start from the init state
        # again: re-register (ADAPTER; if the estimator exposes a reset to a given pose, use it instead).
        # The saved init_pose stays the one written for frame init; the re-registered pose only seeds tracking.
        if order.start != init + 1:
            rgb, depth = rgb_depth(init)
            est.register(K=K_fp, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)  # ADAPTER
        for i in order:
            rgb, depth = rgb_depth(i)
            pose = est.track_one(rgb=rgb, depth=depth, K=K_fp, iteration=2)  # ADAPTER
            T_gun[i] = np.asarray(pose, dtype=np.float64).reshape(4, 4)
            gun_valid[i] = True
finally:
    os.chdir(prev_cwd)

# Tracking is only trustworthy while the gun is visible. Apply the range judged from overlay.mp4.
first, last = GUN_VALID_RANGE
if first is None and last is None:
    print(f"WARNING: ALL {N} frames marked valid; set GUN_VALID_RANGE after checking overlay.mp4")
else:
    lo = 0 if first is None else first
    hi = N - 1 if last is None else last
    outside = (np.arange(N) < lo) | (np.arange(N) > hi)
    gun_valid[outside] = False
    T_gun[outside] = np.eye(4)
print("gun valid on", int(gun_valid.sum()), "of", N, "frames")
save_gun(CLIP / "gun.npz", {"T_cam_gun": T_gun, "valid": gun_valid, "score": gun_valid.astype(float)})
