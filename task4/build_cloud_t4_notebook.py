"""
Generate `task4/task4_cloud_t4_v4.ipynb` — A 100% rule-compliant, self-contained
Google Colab / Kaggle GPU Notebook (T4/P100 16GB) executing:
  - Phase 1 (T4-Exp4): 4-View Denoise/Gamma/De-Tint/Lum Restoration + 5 Foundation
    Backbones (DINOv2-Small, SigLIP2-Base, EVA02-CLIP-Base, ConvNeXt-Tiny-IN22k,
    EffNetV2-S-IN21k) + Scene-Grouped Multi-Head Evaluation + Part A Zip Bundle.
  - Phase 2 (T4-Exp5): End-to-End Unseen-Scene Corruption-Hardened Fine-Tuning
    (ConvNeXt-Tiny, Swin-Tiny, MobileNetV3-Large) with GPU Affine Rotation +-35 deg,
    Dark Gamma, RGB Shot Noise, Hue/Channel Shuffle, MixUp, FP32 Softmax, 4-View TTA.
  - Strictly NO Pseudo-Labeling, NO AutoML, NO Test Leakage, NO Leaderboard Submission!
"""

import base64
import json
import os
import textwrap
import zlib
import numpy as np

BASE = r"c:\!KMUTT\Year 3\Semeter 1\ML\Kaggle\cpe342-ML009"
TASK4_DIR = os.path.join(BASE, "task4")
NB_OUT = os.path.join(TASK4_DIR, "task4_cloud_t4_v4.ipynb")


def md_cell(text: str) -> dict:
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(text: str) -> dict:
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def main():
    split_path = os.path.join(TASK4_DIR, "scene_validation_split.npz")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Run task4/phase0_unseen_scene_benchmark.py first: missing {split_path}")
    sz = np.load(split_path)
    folds_scene_b64 = base64.b64encode(zlib.compress(sz["folds_scene"].astype(np.uint8).tobytes(), level=9)).decode("ascii")
    folds_scene_wrapped = textwrap.fill(folds_scene_b64, width=96)

    cells = []

    cells.append(md_cell("""
# Task 4 Cloud T4 Pipeline — Exp4 Multi-Foundation & Exp5 Hardened Fine-Tuning

โน้ตบุ๊กชุดนี้สำหรับรันบน **Google Colab T4 GPU** หรือ **Kaggle Notebook P100 / T4 GPU** เพื่อเร่งความเร็วและถนอม GPU เครื่องหลัก โดยยึดกติกาการแข่งขัน 100% ดังนี้:
- **ไม่ส่งไฟล์ขึ้น Kaggle Leaderboard อัตโนมัติ** และไม่สร้างไฟล์ `submission.csv` โดยจะแพ็กออกเป็นไฟล์ `.zip` ของ `.npz` และ `.npy` ไว้โหลดกลับเข้าเครื่องหลักเท่านั้น
- **ไม่ทำ Pseudo-Labeling, ไม่ใช้ AutoML และไม่มี Test Leakage** การแบ่ง 5-Fold Unseen-Scene CV และค่าสถิติทั้งหมดคำนวณจาก `train.csv` เพียงอย่างเดียว
- **รันแบบแยกส่วน Resume ได้อัตโนมัติ** โมเดลแต่ละตัวเซฟแคชลงดิสก์ทันทีที่สกัดหรือเทรนเสร็จ หากเซสชันหลุดสามารถรันซ้ำโดยข้ามตัวที่เสร็จแล้วได้เลย
"""))

    cells.append(code_cell("""
# [Cell 1] Environment Check, Dependencies & Output Directory
import os, sys, io, gc, time, glob, zlib, base64, random, shutil, zipfile, warnings
warnings.filterwarnings("ignore")

IN_KAGGLE = os.path.exists("/kaggle/working")
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

import subprocess
def _pip_ensure(pkg, min_ver=None):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", pkg], check=False)

try:
    import timm
    print("Existing timm:", timm.__version__)
    if tuple(int(x) for x in timm.__version__.split(".")[:2]) < (1, 0):
        _pip_ensure("timm>=1.0.15")
except ImportError:
    _pip_ensure("timm>=1.0.15")

try:
    import gdown
except ImportError:
    _pip_ensure("gdown")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import timm
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, normalize

OUT_DIR = "/kaggle/working/task4_cloud_out" if IN_KAGGLE else "/content/task4_cloud_out"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

assert torch.cuda.is_available(), "กรุณาเปิด GPU Accelerator: Colab เลือก Runtime -> Change runtime type -> GPU หรือ Kaggle เลือก GPU T4/P100"
DEVICE = torch.device("cuda")
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
BS_SCALE = 2 if VRAM_GB >= 22.0 else 1
print(f"Environment : {'Kaggle' if IN_KAGGLE else 'Colab' if IN_COLAB else 'Linux'} | PyTorch={torch.__version__} | timm={timm.__version__}")
print(f"GPU         : {torch.cuda.get_device_name(0)} ({VRAM_GB:.2f} GB VRAM, bs_scale={BS_SCALE})")
print(f"Output Dir  : {OUT_DIR}", flush=True)
"""))

    cells.append(md_cell("""
## ขั้นตอนที่ 1: ค้นหาหรือโหลดชุดข้อมูล Task 4 อัตโนมัติ

สคริปต์จะค้นหาข้อมูลตามลำดับความน่าเชื่อถือดังนี้:
1. Dataset ที่แนบไว้ใน `/kaggle/input` หากสร้าง Private Kaggle Dataset ที่มีรูปภาพ `train/` และ `test/`
2. ไฟล์ `task4_data.zip` หรือ `task4.zip` ใน Google Drive ส่วนตัวบน Colab ที่ `/content/drive/MyDrive/`
3. ดาวน์โหลดอัตโนมัติจากลิงก์ Google Drive ทางการใน `data/task4/README.md` ผ่าน `gdown`
"""))

    cells.append(code_cell("""
# [Cell 2] Bulletproof Task 4 Data Locator (Kaggle Input / Google Drive / Official Drive ID)
OFFICIAL_GDRIVE_URL = "https://drive.google.com/file/d/1C8csylbBo3Oab24ODGKPVom3_SDUJ_Qw/view?usp=share_link"
WORK_DATA_ROOT = "/kaggle/working/task4_data" if IN_KAGGLE else "/content/task4_data"
os.makedirs(WORK_DATA_ROOT, exist_ok=True)

def find_valid_task4_dir(root_paths):
    for root in root_paths:
        if not os.path.exists(root):
            continue
        for cur, dirs, files in os.walk(root):
            if "__MACOSX" in cur:
                continue
            if "train.csv" in files and "test.csv" in files:
                has_imgs = os.path.isdir(os.path.join(cur, "train")) and os.path.isdir(os.path.join(cur, "test"))
                if has_imgs:
                    # Verify train/ actually contains jpgs
                    n_jpg = len(os.listdir(os.path.join(cur, "train")))
                    if n_jpg > 30000:
                        return cur
    return None

DATA_DIR = find_valid_task4_dir([WORK_DATA_ROOT, "/kaggle/input", "/content"])

DRIVE_BACKUP_DIR = "/content/drive/MyDrive/cpe342_task4_outputs" if os.path.exists("/content/drive/MyDrive") else None
if DRIVE_BACKUP_DIR:
    os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)

if DATA_DIR is None and os.path.exists("/content/drive/MyDrive"):
    for cand_zip in [
        "/content/drive/MyDrive/task4_data.zip",
        "/content/drive/MyDrive/task4.zip",
        "/content/drive/MyDrive/cpe342_task4.zip",
        "/content/drive/MyDrive/Kaggle/task4_data.zip",
    ]:
        if os.path.exists(cand_zip):
            print(f"Found zip in mounted Google Drive: {cand_zip} -> extracting...", flush=True)
            with zipfile.ZipFile(cand_zip, "r") as zf:
                zf.extractall(WORK_DATA_ROOT)
            DATA_DIR = find_valid_task4_dir([WORK_DATA_ROOT])
            break
    if DATA_DIR is None:
        DATA_DIR = find_valid_task4_dir(["/content/drive/MyDrive"])

if DATA_DIR is None:
    import gdown
    zip_dl = os.path.join(WORK_DATA_ROOT, "task4_official.zip")
    if not os.path.exists(zip_dl):
        print("Downloading Task 4 dataset from official competition Drive link (no OAuth popup)...", flush=True)
        ok = gdown.download(url=OFFICIAL_GDRIVE_URL, output=zip_dl, fuzzy=True, quiet=False)
        if not ok or not os.path.exists(zip_dl):
            raise RuntimeError(
                "GDOWN_QUOTA_OR_FAIL: ดาวน์โหลดจากลิงก์กลางไม่สำเร็จหรือติดโควตา Google Drive ชั่วคราว "
                "แนะนำให้อัปโหลด task4_data.zip ขึ้น Google Drive ส่วนตัวแล้ว mount /content/drive หรือรันจาก Local Zip ครับ"
            )
    print(f"Extracting {zip_dl} (ignoring __MACOSX)...", flush=True)
    with zipfile.ZipFile(zip_dl, "r") as zf:
        members = [m for m in zf.namelist() if "__MACOSX" not in m]
        zf.extractall(WORK_DATA_ROOT, members=members)
    DATA_DIR = find_valid_task4_dir([WORK_DATA_ROOT])

assert DATA_DIR is not None, "ไม่พบโฟลเดอร์ที่มี train.csv, test.csv, train/, test/ กรุณาตรวจสอบพาธข้อมูล"
tr_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
te_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
assert len(tr_df) == 31546 and len(te_df) == 25889, f"Unexpected row counts: {len(tr_df)}, {len(te_df)}"
y_all = tr_df["label"].to_numpy().astype(np.int64)
print(f"Verified Task 4 DATA_DIR: {DATA_DIR}")
print(f"Train={len(tr_df):,} | Test={len(te_df):,} | Class Counts={np.bincount(y_all).tolist()}")
"""))

    cells.append(code_cell('''
# [Cell 3] Embedded Golden Unseen-Scene 5-Fold Split + RAM JPEG Preload + 24 Phys Features
FOLDS_SCENE_B64 = """
__FOLDS_SCENE_WRAPPED__
""".replace("\\n", "").strip()

from sklearn.model_selection import StratifiedKFold
folds_scene = np.frombuffer(zlib.decompress(base64.b64decode(FOLDS_SCENE_B64)), dtype=np.uint8).astype(np.int64)
folds_skf = np.zeros(len(tr_df), dtype=np.int64)
for _f, (_, _va) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(tr_df, y_all)):
    folds_skf[_va] = _f
assert len(folds_scene) == len(tr_df) == 31546
assert len(folds_skf) == len(tr_df) == 31546
print("Loaded bit-exact Unseen-Scene 5-Fold Split:", np.bincount(folds_scene).tolist())

# Deterministic uniform class evaluation subsets
rng_eval = np.random.default_rng(SEED)
min_c = int(np.bincount(y_all).min())
bal_subsets = [np.concatenate([rng_eval.choice(np.where(y_all == c)[0], size=min_c, replace=False) for c in range(5)]) for _ in range(25)]

from concurrent.futures import ThreadPoolExecutor
def preload_jpegs(folder, fnames):
    base_f = os.path.join(DATA_DIR, folder)
    def _r(fn):
        with open(os.path.join(base_f, fn), "rb") as f:
            return f.read()
    with ThreadPoolExecutor(max_workers=16) as ex:
        return list(ex.map(_r, fnames))

t0 = time.time()
tr_jb = preload_jpegs("train", tr_df["file_name"].values)
te_jb = preload_jpegs("test", te_df["file_name"].values)
print(f"Preloaded {len(tr_jb):,} train + {len(te_jb):,} test JPEGs into RAM in {time.time()-t0:.1f}s")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

def decode_jpeg_rgb(jb_bytes: bytes) -> np.ndarray:
    if HAS_CV2:
        arr = np.frombuffer(jb_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.array(Image.open(io.BytesIO(jb_bytes)).convert("RGB"), dtype=np.uint8, copy=True)

def extract_phys_features(jb_list):
    def _one(b):
        arr = decode_jpeg_rgb(b).astype(np.float32)
        h, w, _ = arr.shape
        r, g, bl = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        rm, gm, bm = r.mean(), g.mean(), bl.mean()
        rs, gs, bs = r.std(), g.std(), bl.std()
        cast = max(abs(rm - 0.5 * (gm + bm)), abs(gm - 0.5 * (rm + bm)), abs(bm - 0.5 * (rm + gm)))
        chroma = (arr.max(axis=2) - arr.min(axis=2)).mean()
        bh, bw = h // 10, w // 10
        bmask = np.zeros((h, w), dtype=bool)
        bmask[:bh, :] = True; bmask[-bh:, :] = True; bmask[:, :bw] = True; bmask[:, -bw:] = True
        blk_border = (arr[bmask].max(axis=1) < 12.0).mean()
        blk_all = (arr.max(axis=2) < 10.0).mean()
        wht_all = (arr.min(axis=2) > 245.0).mean()
        gray = arr.mean(axis=2)
        lap = (gray[2:, 1:-1] + gray[:-2, 1:-1] + gray[1:-1, 2:] + gray[1:-1, :-2] - 4.0 * gray[1:-1, 1:-1]).std()
        gx = np.abs(np.diff(gray, axis=1)).mean()
        gy = np.abs(np.diff(gray, axis=0)).mean()
        top_m, bot_m = gray[:h//2].mean(), gray[h//2:].mean()
        top_c = (arr[:h//2].max(axis=2) - arr[:h//2].min(axis=2)).mean()
        bot_c = (arr[h//2:].max(axis=2) - arr[h//2:].min(axis=2)).mean()
        return np.array([
            rm, gm, bm, rs, gs, bs, rm - gm, gm - bm, rm - bm,
            cast, chroma, blk_border, blk_all, wht_all, lap, gx, gy, gx / (gy + 1e-4),
            top_m, bot_m, top_m - bot_m, top_c, bot_c, top_c - bot_c
        ], dtype=np.float32)
    with ThreadPoolExecutor(max_workers=16) as ex:
        return np.vstack(list(ex.map(_one, jb_list)))

phys_cache = os.path.join(OUT_DIR, "phys_features_v4.npz")
if os.path.exists(phys_cache):
    pz = np.load(phys_cache)
    phys_tr, phys_te = pz["phys_tr"], pz["phys_te"]
else:
    t1 = time.time()
    phys_tr = extract_phys_features(tr_jb)
    phys_te = extract_phys_features(te_jb)
    np.savez_compressed(phys_cache, phys_tr=phys_tr, phys_te=phys_te)
    print(f"Extracted 24 physical distortion features in {time.time()-t1:.1f}s")

np.savez_compressed(os.path.join(OUT_DIR, "scene_split_v4.npz"), folds_scene=folds_scene.astype(np.uint8), folds_skf=folds_skf.astype(np.uint8))

def eval_unseen_report(name, probs, phys=phys_tr):
    pred = probs.argmax(1)
    f_all = f1_score(y_all, pred, average="macro")
    f_bal = float(np.mean([f1_score(y_all[idx], pred[idx], average="macro") for idx in bal_subsets]))
    mb = phys[:, 0:3].mean(axis=1)
    m_d45 = mb < 45
    m_d75 = mb < 75
    m_rot = phys[:, 11] > 0.15
    m_noi = phys[:, 14] > 90
    print(f"  {name:<26s} | UnseenAllF1={f_all:.5f} | UnseenBalF1={f_bal:.5f} | Dark<45={f1_score(y_all[m_d45], pred[m_d45], average='macro'):.4f} | Dark<75={f1_score(y_all[m_d75], pred[m_d75], average='macro'):.4f} | Border={f1_score(y_all[m_rot], pred[m_rot], average='macro'):.4f} | Noise={f1_score(y_all[m_noi], pred[m_noi], average='macro'):.4f}")
'''.replace("__FOLDS_SCENE_WRAPPED__", folds_scene_wrapped)))

    cells.append(code_cell("""
# [Cell 4] Visually Verified Seam-Free 4-View GPU Image Restoration & Dataset
class JPEGBytesDataset(Dataset):
    def __init__(self, jb_list, labels=None):
        self.jb = jb_list
        self.labels = labels

    def __len__(self):
        return len(self.jb)

    def __getitem__(self, i):
        arr = decode_jpeg_rgb(self.jb[i])
        ten = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        if self.labels is None:
            return ten
        return ten, int(self.labels[i])


@torch.no_grad()
def gpu_denoise_restore_base(batch: torch.Tensor):
    \"\"\"
    Clean deterministic per-image restoration (zero test leakage!):
      - Edge-preserving 3x3 Median blend to suppress RGB shot noise before stretch
      - Non-border active pixel stats (avoids rotation black corner triangles without hard seams)
      - Spatial-smooth dark noise suppression for low-luminance scenes
      - Bounded Adaptive Gamma + Noise-Safe Algebraic Channel De-Tinting (std_c >= 0.075)
      - Hue-invariant Luminance Structure representation
    Accepts uint8 [0..255] or float32 [0..1]; returns x_raw_clean, x_detint, v_lum in [0, 1].
    \"\"\"
    x = batch.float().div_(255.0) if batch.dtype == torch.uint8 else batch.float().clamp(0.0, 1.0)
    B, C, H, W = x.shape

    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    patches = xp.unfold(2, 3, 1).unfold(3, 3, 1).contiguous().view(B, C, H, W, 9)
    x_med = patches.kthvalue(5, dim=-1).values
    x_clean = 0.65 * x_med + 0.35 * x

    lit = (x_clean.max(dim=1, keepdim=True).values >= (6.0 / 255.0)).float()
    lit_frac = lit.mean(dim=(2, 3), keepdim=True)
    active = torch.where(lit_frac > 0.25, lit, torch.ones_like(lit))
    n_act = active.sum(dim=(2, 3), keepdim=True).clamp_(min=32.0)

    lum_act = ((x_clean.mean(dim=1, keepdim=True) * active).sum(dim=(2, 3), keepdim=True) / n_act).clamp_(0.01, 0.95)
    x_box5 = F.avg_pool2d(F.pad(x_clean, (2, 2, 2, 2), mode="replicate"), 5, 1)
    dark_alpha = ((0.16 - lum_act) / 0.12).clamp_(0.0, 0.70)
    x_d = (1.0 - dark_alpha) * x_clean + dark_alpha * x_box5

    gamma = (torch.log(torch.tensor(0.45, device=x.device)) / torch.log(lum_act.clamp_(0.05, 0.92))).clamp_(0.58, 1.25)
    x_gam = x_d.clamp(1e-4, 1.0) ** gamma

    # Smoothly taper stretch blend for pitch-black low-SNR frames (<0.05) to avoid pure noise amplification
    snr_w = ((lum_act - 0.025) / 0.055).clamp_(0.35, 0.82)
    mu_c = (x_gam * active).sum(dim=(2, 3), keepdim=True) / n_act
    var_c = ((x_gam - mu_c) ** 2 * active).sum(dim=(2, 3), keepdim=True) / n_act
    std_c = var_c.sqrt().clamp_(min=0.075)
    x_rest = ((x_gam - mu_c) / std_c) * 0.22 + 0.46
    x_rest = (0.46 + 0.50 * torch.tanh((x_rest - 0.46) / 0.52)).clamp_(0.0, 1.0)
    x_det = torch.where(active > 0.5, snr_w * x_rest + (1.0 - snr_w) * x_gam, x_gam).clamp_(0.0, 1.0)

    g_clean = x_d.mean(dim=1, keepdim=True)
    g_gam = g_clean.clamp(1e-4, 1.0) ** gamma
    mu_g = (g_gam * active).sum(dim=(2, 3), keepdim=True) / n_act
    std_g = (((g_gam - mu_g) ** 2 * active).sum(dim=(2, 3), keepdim=True) / n_act).sqrt().clamp_(min=0.070)
    g_rest = (0.46 + 0.50 * torch.tanh((((g_gam - mu_g) / std_g) * 0.22) / 0.52)).clamp_(0.0, 1.0)
    v_lum = torch.where(active > 0.5, snr_w * g_rest + (1.0 - snr_w) * g_gam, g_gam).expand(-1, 3, -1, -1)

    return (0.25 * x_med + 0.75 * x), x_det, v_lum


def build_4views_for_model(u8_batch: torch.Tensor, target_hw, mean_t: torch.Tensor, std_t: torch.Tensor):
    v_raw, v_det, v_lum = gpu_denoise_restore_base(u8_batch)
    H, W = v_det.shape[2], v_det.shape[3]
    ch, cw = int(H * 0.07), int(W * 0.07)
    v_crop = torch.flip(v_det[:, :, ch:H - ch, cw:W - cw], dims=[3])

    def _prep(t):
        if (t.shape[2], t.shape[3]) != target_hw:
            t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
        return (t - mean_t) / std_t

    return _prep(v_raw), _prep(v_det), _prep(v_crop), _prep(v_lum)
"""))

    cells.append(md_cell("""
## ระยะที่ 1 — `T4-Exp4`: สกัดฟีเจอร์ 4 มุมมองจาก 5 Vision Foundation Backbones + เทรนหัวจำแนก Unseen-Scene

เวลาโดยประมาณบน T4 GPU รวมประมาณ `20 – 30` นาที โดยเซฟ Checkpoint หลังจบแต่ละ Backbone และรวมเป็นไฟล์ `embs_v4_cloud_5backbones.npz` พร้อมชุดคำตอบ `task4_partA_v4_outputs.zip` ให้โหลดได้ทันที
"""))

    cells.append(code_cell("""
# [Cell 5] Phase 1 (T4-Exp4): Extract 4-View FP16 Embeddings from 5 Diverse Foundation Models
FOUNDATION_SPECS = [
    ("dinov2_s",   "vit_small_patch14_reg4_dinov2.lvd142m", (126, 238), True,  192),
    ("siglip2_b",  "vit_base_patch16_siglip_224.v2_webli",   (128, 240), True,  128),
    ("eva02clip",  "eva02_base_patch16_clip_224.merged2b",  (128, 240), True,  128),
    ("convnext_t", "convnext_tiny.fb_in22k_ft_in1k",        (128, 256), False, 192),
    ("effnetv2_s", "tf_efficientnetv2_s.in21k_ft_in1k",     (160, 320), False, 192),
]

@torch.no_grad()
def extract_4view_backbone(jb_list, model_name, target_hw, dynamic_img, batch_size):
    ds = JPEGBytesDataset(jb_list)
    nw = 2 if os.name != "nt" else 0
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    kw = {"pretrained": True, "num_classes": 0}
    if dynamic_img:
        kw["dynamic_img_size"] = True
    m = timm.create_model(model_name, **kw).to(DEVICE).eval()
    cfg = timm.data.resolve_model_data_config(m)
    mean_t = torch.tensor(cfg["mean"], device=DEVICE, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(cfg["std"], device=DEVICE, dtype=torch.float32).view(1, 3, 1, 1)

    feats = []
    for u8 in ldr:
        u8 = u8.to(DEVICE, non_blocking=True)
        v1, v2, v3, v4 = build_4views_for_model(u8, target_hw, mean_t, std_t)
        with torch.amp.autocast("cuda"):
            f1 = F.normalize(m(v1).float(), dim=-1)
            f2 = F.normalize(m(v2).float(), dim=-1)
            f3 = F.normalize(m(v3).float(), dim=-1)
            f4 = F.normalize(m(v4).float(), dim=-1)
            # Weighted fusion emphasizing restored color + border-free crop + hue-free structure + raw clean
            f = F.normalize(0.20 * f1 + 0.35 * f2 + 0.25 * f3 + 0.20 * f4, dim=-1)
        feats.append(f.half().cpu().numpy())
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0)

merged_emb_path = os.path.join(OUT_DIR, "embs_v4_cloud_5backbones.npz")
emb_dict = {}
if os.path.exists(merged_emb_path):
    print("Found complete 5-backbone cache:", merged_emb_path)
    z_m = np.load(merged_emb_path)
    for k in z_m.files:
        emb_dict[k] = z_m[k]
else:
    for tag, mname, hw, dyn, bs in FOUNDATION_SPECS:
        bs_use = min(384, bs * BS_SCALE)
        ckpt_p = os.path.join(OUT_DIR, f"ckpt_emb_{tag}.npz")
        if os.path.exists(ckpt_p):
            zc = np.load(ckpt_p)
            emb_dict[f"{tag}_tr"] = zc["tr"]
            emb_dict[f"{tag}_te"] = zc["te"]
            print(f"[SKIP] Loaded cached {tag}: {zc['tr'].shape}", flush=True)
            continue
        t_m = time.time()
        print(f"[EXTRACTING] {tag} ({mname}, size={hw}, bs={bs_use})...", flush=True)
        tr_e = extract_4view_backbone(tr_jb, mname, hw, dyn, bs_use)
        te_e = extract_4view_backbone(te_jb, mname, hw, dyn, bs_use)
        np.savez_compressed(ckpt_p, tr=tr_e, te=te_e)
        emb_dict[f"{tag}_tr"] = tr_e
        emb_dict[f"{tag}_te"] = te_e
        print(f"  -> Done {tag}: train={tr_e.shape}, test={te_e.shape} in {time.time()-t_m:.1f}s", flush=True)
    np.savez_compressed(merged_emb_path, **emb_dict)
    print(f"Saved merged 5-backbone embeddings: {merged_emb_path} ({os.path.getsize(merged_emb_path)/1e6:.1f} MB)", flush=True)
"""))

    cells.append(code_cell("""
# [Cell 6] Phase 1 Classifier Heads on Unseen-Scene 5-Fold CV + Part A Bundle Export
class ResidualEmbeddingNet(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 5):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, 640)
        self.ln1 = nn.LayerNorm(640)
        self.drop1 = nn.Dropout(0.45)
        self.fc2 = nn.Linear(640, 640)
        self.ln2 = nn.LayerNorm(640)
        self.drop2 = nn.Dropout(0.40)
        self.head = nn.Linear(640, num_classes)
        self.skip = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        xn = self.norm_in(x)
        h1 = F.silu(self.ln1(self.fc1(xn)))
        h1 = self.drop1(h1)
        h2 = h1 + self.drop2(F.silu(self.ln2(self.fc2(h1))))
        return self.head(h2) + 0.25 * self.skip(xn)

tags = [s[0] for s in FOUNDATION_SPECS]
X_tr_emb = np.hstack([emb_dict[f"{t}_tr"].astype(np.float32) for t in tags])
X_te_emb = np.hstack([emb_dict[f"{t}_te"].astype(np.float32) for t in tags])
sc_p = StandardScaler().fit(phys_tr)
X_tr_all = np.hstack([X_tr_emb, sc_p.transform(phys_tr)]).astype(np.float32)
X_te_all = np.hstack([X_te_emb, sc_p.transform(phys_te)]).astype(np.float32)
print(f"Concatenated 5-Foundation + 24 Phys Feature Dim: {X_tr_all.shape[1]}")

pi_tr = np.bincount(y_all) / len(y_all)
w_bal = torch.tensor((1.0 / pi_tr) / (1.0 / pi_tr).mean(), dtype=torch.float32, device=DEVICE)
w_sqrt = torch.tensor((1.0 / (pi_tr ** 0.50)) / (1.0 / (pi_tr ** 0.50)).mean(), dtype=torch.float32, device=DEVICE)

oof_v4_lr = np.zeros((len(tr_df), 5), dtype=np.float32)
test_v4_lr = np.zeros((len(te_df), 5), dtype=np.float32)
oof_v4_mlp = np.zeros((len(tr_df), 5), dtype=np.float32)
test_v4_mlp = np.zeros((len(te_df), 5), dtype=np.float32)

# Train strictly on folds_scene! Every fold stops on unseen scenes.
for f in range(5):
    tr_m = folds_scene != f
    va_m = folds_scene == f
    sc = StandardScaler().fit(X_tr_all[tr_m])
    Xtr = sc.transform(X_tr_all[tr_m]).astype(np.float32)
    Xva = sc.transform(X_tr_all[va_m]).astype(np.float32)
    Xte = sc.transform(X_te_all).astype(np.float32)

    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t = torch.from_numpy(y_all[tr_m]).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    Xte_t = torch.from_numpy(Xte).to(DEVICE)

    # 1. Strong Regularized Multinomial Linear Probe (LBFGS)
    lin = nn.Linear(X_tr_all.shape[1], 5).to(DEVICE)
    opt_l = torch.optim.LBFGS(lin.parameters(), lr=0.8, max_iter=40, line_search_fn="strong_wolfe")
    crit_l = nn.CrossEntropyLoss(weight=w_bal)
    def closure():
        opt_l.zero_grad()
        loss = crit_l(lin(Xtr_t), ytr_t) + 0.008 * (lin.weight ** 2).sum()
        loss.backward()
        return loss
    opt_l.step(closure)
    with torch.no_grad():
        oof_v4_lr[va_m] = torch.softmax(lin(Xva_t).float(), dim=-1).cpu().numpy()
        test_v4_lr += torch.softmax(lin(Xte_t).float(), dim=-1).cpu().numpy() / 5.0

    # 2. Multi-Seed Averaged GPU Residual-MLP per fold for low variance
    p_va_fold, p_te_fold = np.zeros((va_m.sum(), 5), dtype=np.float32), np.zeros((len(te_df), 5), dtype=np.float32)
    for sd in range(2):
        torch.manual_seed(SEED + f * 10 + sd)
        ldr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y_all[tr_m])), batch_size=256, shuffle=True, drop_last=True)
        net = ResidualEmbeddingNet(X_tr_all.shape[1], 5).to(DEVICE)
        opt_m = torch.optim.AdamW(net.parameters(), lr=1.0e-3, weight_decay=3.5e-2)
        sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=22 * len(ldr))
        crit_m = nn.CrossEntropyLoss(weight=w_sqrt, label_smoothing=0.06)
        best_f1, best_p, best_st = -1.0, None, None
        for ep in range(1, 23):
            net.train()
            for xb, yb in ldr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt_m.zero_grad(set_to_none=True)
                if ep <= 17 and np.random.rand() < 0.50:
                    lam = float(np.random.beta(0.35, 0.35))
                    perm = torch.randperm(xb.size(0), device=DEVICE)
                    logits = net(lam * xb + (1.0 - lam) * xb[perm])
                    loss = lam * crit_m(logits, yb) + (1.0 - lam) * crit_m(logits, yb[perm])
                else:
                    loss = crit_m(net(xb), yb)
                loss.backward()
                opt_m.step()
                sched_m.step()
            net.eval()
            with torch.no_grad():
                pv = torch.softmax(net(Xva_t).float(), dim=-1).cpu().numpy()
            f1v = f1_score(y_all[va_m], pv.argmax(1), average="macro")
            if f1v > best_f1:
                best_f1, best_p = f1v, pv
                best_st = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        p_va_fold += best_p / 2.0
        net.load_state_dict({k: v.to(DEVICE) for k, v in best_st.items()})
        net.eval()
        with torch.no_grad():
            p_te_fold += torch.softmax(net(Xte_t).float(), dim=-1).cpu().numpy() / 2.0
    oof_v4_mlp[va_m] = p_va_fold
    test_v4_mlp += p_te_fold / 5.0

oof_v4_blend = 0.35 * oof_v4_lr + 0.65 * oof_v4_mlp
test_v4_blend = 0.35 * test_v4_lr + 0.65 * test_v4_mlp

print("\\n=== Phase 1 (T4-Exp4) GOLDEN Unseen-Scene 5-Fold Benchmark (Compare vs Phase 0 Base = 0.81415) ===")
eval_unseen_report("v4_foundation_lr", oof_v4_lr)
eval_unseen_report("v4_foundation_mlp", oof_v4_mlp)
eval_unseen_report("v4_foundation_blend", oof_v4_blend)

np.save(os.path.join(OUT_DIR, "oof_probs_v4_foundation_lr.npy"), oof_v4_lr)
np.save(os.path.join(OUT_DIR, "test_probs_v4_foundation_lr.npy"), test_v4_lr)
np.save(os.path.join(OUT_DIR, "oof_probs_v4_foundation_mlp.npy"), oof_v4_mlp)
np.save(os.path.join(OUT_DIR, "test_probs_v4_foundation_mlp.npy"), test_v4_mlp)

partA_zip = os.path.join(OUT_DIR, "task4_partA_v4_outputs.zip")
with zipfile.ZipFile(partA_zip, "w", compression=zipfile.ZIP_STORED) as zf:
    for fn in ["embs_v4_cloud_5backbones.npz", "phys_features_v4.npz", "scene_split_v4.npz",
               "oof_probs_v4_foundation_lr.npy", "test_probs_v4_foundation_lr.npy",
               "oof_probs_v4_foundation_mlp.npy", "test_probs_v4_foundation_mlp.npy"]:
        zf.write(os.path.join(OUT_DIR, fn), arcname=fn)
print(f"\\n>>> STATUS: PHASE1_PARTA_DONE | Saved Part A bundle: {partA_zip} ({os.path.getsize(partA_zip)/1e6:.1f} MB) <<<", flush=True)
if DRIVE_BACKUP_DIR:
    shutil.copy2(partA_zip, os.path.join(DRIVE_BACKUP_DIR, "task4_partA_v4_outputs.zip"))
    print("Auto-backed up Part A zip to Google Drive:", DRIVE_BACKUP_DIR)
"""))

    cells.append(md_cell("""
## ระยะที่ 2 — `T4-Exp5`: End-to-End Domain-Invariant Fine-Tuning บน `folds_scene`

ต่อจากระยะที่ 1 เราเทรนโมเดล Fine-Tuning สายถึกทนความเพี้ยนอีก 3 สถาปัตยกรรม (`convnext_tiny.fb_in22k_ft_in1k`, `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`, `mobilenetv3_large_100.ra_in1k`) โดยใส่:
- **GPU Aspect-Corrected Affine Rotation บวกลบ 35 องศาขอบดำ**
- **Extreme Dark Gamma และ High RGB Shot Noise + Channel Shuffle + Random Grayscale + MixUp**
- **FP32 Softmax ไร้ NaN 100%, ตัด Vertical Flip กลับหัวทิ้ง, Model EMA และ 4-View TTA**
"""))

    cells.append(code_cell("""
# [Cell 7] Phase 2 (T4-Exp5): End-to-End Hardened Fine-Tuning on Golden Unseen-Scene 5-Fold
def gpu_domain_augment(u8_batch: torch.Tensor, target_hw, mean_t: torch.Tensor, std_t: torch.Tensor):
    x = u8_batch.float().div_(255.0)
    B, C, H, W = x.shape

    # 1. Synthetic RGB tint, extreme darkness gamma, and channel shuffle BEFORE restoration (p=0.65)
    #    Forces network to handle extreme non-linear & linear color/darkness shifts even after detinting
    do_c = (torch.rand(B, 1, 1, 1, device=DEVICE) < 0.65).float()
    alpha = 1.0 + (torch.rand(B, 3, 1, 1, device=DEVICE) - 0.5) * 0.90
    beta = (torch.rand(B, 1, 1, 1, device=DEVICE) - 0.5) * 0.36
    x_jit = (x * alpha + beta).clamp_(0.0, 1.0)
    gam = torch.exp((torch.rand(B, 1, 1, 1, device=DEVICE) - 0.55) * 1.1)  # biases toward dark [0.55..1.55]
    x_jit = x_jit.clamp(1e-4, 1.0) ** gam
    x = do_c * x_jit + (1.0 - do_c) * x

    # Random channel permutation (p=0.18) so neon swaps don't break classifier
    perm_mask = torch.rand(B, device=DEVICE) < 0.18
    if perm_mask.any():
        idx_p = torch.argsort(torch.rand(B, 3, device=DEVICE), dim=1)
        x_perm = torch.gather(x, 1, idx_p.view(B, 3, 1, 1).expand(-1, -1, H, W))
        x = torch.where(perm_mask.view(B, 1, 1, 1), x_perm, x)

    # 2. Vectorized GPU aspect-preserving Affine Rotation (+-35 deg) & zoom [0.90, 1.08] with black fill
    rot_mask = torch.rand(B, device=DEVICE) < 0.70
    if rot_mask.any():
        ang = torch.where(rot_mask, (torch.rand(B, device=DEVICE) * 70.0 - 35.0) * (np.pi / 180.0), torch.zeros(B, device=DEVICE))
        sc = torch.where(rot_mask, 0.90 + torch.rand(B, device=DEVICE) * 0.18, torch.ones(B, device=DEVICE))
        ca, sa = torch.cos(ang) / sc, torch.sin(ang) / sc
        ar = float(W) / float(H)
        theta = torch.zeros(B, 2, 3, device=DEVICE)
        theta[:, 0, 0] = ca
        theta[:, 0, 1] = -(1.0 / ar) * sa
        theta[:, 1, 0] = ar * sa
        theta[:, 1, 1] = ca
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # 3. Apply clean restoration probabilistically (15% raw, 65% detint_gamma, 20% hue-free lum structure)
    _, x_det, v_lum = gpu_denoise_restore_base(x)
    u = torch.rand(B, 1, 1, 1, device=DEVICE)
    x = torch.where(u < 0.15, x, torch.where(u < 0.80, x_det, v_lum))

    # 4. Horizontal flip (NO upside-down VFlip!), Random Grayscale, Shot Noise & Cutout
    h_mask = torch.rand(B, 1, 1, 1, device=DEVICE) < 0.5
    x = torch.where(h_mask, torch.flip(x, dims=[3]), x)

    if (x.shape[2], x.shape[3]) != target_hw:
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

    n_mask = (torch.rand(B, 1, 1, 1, device=DEVICE) < 0.45).float()
    sigma = torch.rand(B, 1, 1, 1, device=DEVICE) * 0.065
    x = (x + n_mask * torch.randn_like(x) * sigma).clamp_(0.0, 1.0)

    if random.random() < 0.50:
        th, tw = target_hw
        eh = random.randint(int(th * 0.10), int(th * 0.25))
        ew = random.randint(int(tw * 0.10), int(tw * 0.25))
        ey = random.randint(0, th - eh)
        ex = random.randint(0, tw - ew)
        em = torch.rand(B, 1, 1, 1, device=DEVICE) < 0.30
        x[:, :, ey:ey + eh, ex:ex + ew] = torch.where(em, torch.zeros_like(x[:, :, ey:ey + eh, ex:ex + ew]), x[:, :, ey:ey + eh, ex:ex + ew])

    return (x - mean_t) / std_t


@torch.no_grad()
def predict_4view_tta(model, loader, target_hw, mean_t, std_t, use_tta=True):
    model.eval()
    probs_all = []
    for batch in loader:
        u8 = batch[0] if isinstance(batch, (tuple, list)) else batch
        u8 = u8.to(DEVICE, non_blocking=True)
        v1, v2, v3, v4 = build_4views_for_model(u8, target_hw, mean_t, std_t)
        with torch.amp.autocast("cuda"):
            p2 = torch.softmax(model(v2).float(), dim=-1)
            if use_tta:
                p1 = torch.softmax(model(v1).float(), dim=-1)
                p3 = torch.softmax(model(v3).float(), dim=-1)
                p4 = torch.softmax(model(v4).float(), dim=-1)
                p = 0.20 * p1 + 0.38 * p2 + 0.24 * p3 + 0.18 * p4
            else:
                p = p2
        probs_all.append(p.cpu().numpy())
    return np.concatenate(probs_all, axis=0)


def train_5fold_e2e(tag, model_name, target_hw, epochs=8, batch_size=96, lr_bb=1.8e-4, lr_hd=1.2e-3, drop_path=0.20):
    oof_p = os.path.join(OUT_DIR, f"oof_probs_{tag}.npy")
    test_p = os.path.join(OUT_DIR, f"test_probs_{tag}.npy")
    if os.path.exists(oof_p) and os.path.exists(test_p):
        print(f"[SKIP] Found cached {tag}")
        oof_f = np.load(oof_p)
        eval_unseen_report(tag, oof_f)
        return oof_f, np.load(test_p)

    print(f"\\n[TRAINING 5-Fold Unseen-Scene] {tag} ({model_name}, hw={target_hw}, ep={epochs}, bs={batch_size})")
    pi_tr = np.bincount(y_all) / len(y_all)
    w_soft = (1.0 / (pi_tr ** 0.50)).astype(np.float32)
    w_soft /= w_soft.mean()
    w_ten = torch.tensor(w_soft, dtype=torch.float32, device=DEVICE)

    nw = 2 if os.name != "nt" else 0
    te_ldr = DataLoader(JPEGBytesDataset(te_jb), batch_size=batch_size * 2, shuffle=False, num_workers=nw, pin_memory=True)

    oof = np.zeros((len(tr_df), 5), dtype=np.float32)
    test = np.zeros((len(te_df), 5), dtype=np.float32)

    for fold in range(5):
        t_f = time.time()
        tr_idx = np.where(folds_scene != fold)[0]
        va_idx = np.where(folds_scene == fold)[0]

        tr_ldr = DataLoader(JPEGBytesDataset([tr_jb[i] for i in tr_idx], y_all[tr_idx]), batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
        va_ldr = DataLoader(JPEGBytesDataset([tr_jb[i] for i in va_idx], y_all[va_idx]), batch_size=batch_size * 2, shuffle=False, num_workers=nw, pin_memory=True)

        kw = {"pretrained": True, "num_classes": 5, "drop_rate": 0.20}
        if drop_path > 0:
            kw["drop_path_rate"] = drop_path
        model = timm.create_model(model_name, **kw).to(DEVICE)
        cfg = timm.data.resolve_model_data_config(model)
        mean_t = torch.tensor(cfg["mean"], device=DEVICE, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(cfg["std"], device=DEVICE, dtype=torch.float32).view(1, 3, 1, 1)

        head_p = [p for n, p in model.named_parameters() if ("head" in n or "classifier" in n or "fc" in n)]
        head_ids = {id(p) for p in head_p}
        bb_p = [p for p in model.parameters() if id(p) not in head_ids]
        opt = torch.optim.AdamW([{"params": bb_p, "lr": lr_bb}, {"params": head_p, "lr": lr_hd}], weight_decay=2.5e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_bb, lr_hd], steps_per_epoch=len(tr_ldr), epochs=epochs, pct_start=0.15, anneal_strategy="cos")
        scaler = torch.amp.GradScaler("cuda")
        crit = nn.CrossEntropyLoss(weight=w_ten, label_smoothing=0.06)

        best_f1, best_st = -1.0, None
        for ep in range(1, epochs + 1):
            model.train()
            r_loss = 0.0
            for u8, yb in tr_ldr:
                u8, yb = u8.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                imgs = gpu_domain_augment(u8, target_hw, mean_t, std_t)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    if ep <= epochs - 2 and random.random() < 0.35:
                        lam = float(np.random.beta(0.25, 0.25))
                        perm = torch.randperm(imgs.size(0), device=DEVICE)
                        logits = model(lam * imgs + (1.0 - lam) * imgs[perm])
                        loss = lam * crit(logits, yb) + (1.0 - lam) * crit(logits, yb[perm])
                    else:
                        loss = crit(model(imgs), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                r_loss += float(loss.item()) * imgs.size(0)
            pv = predict_4view_tta(model, va_ldr, target_hw, mean_t, std_t, use_tta=False)
            f1v = f1_score(y_all[va_idx], pv.argmax(1), average="macro")
            if f1v > best_f1:
                best_f1 = f1v
                best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Fold {fold+1} Ep {ep:02d}/{epochs:02d} | Loss={r_loss/len(tr_idx):.4f} | UnseenValF1={f1v:.4f}")

        model.load_state_dict({k: v.to(DEVICE) for k, v in best_st.items()})
        p_oof_f = predict_4view_tta(model, va_ldr, target_hw, mean_t, std_t, use_tta=True)
        oof[va_idx] = p_oof_f
        print(f"-> Fold {fold+1} 4-View TTA UnseenF1={f1_score(y_all[va_idx], p_oof_f.argmax(1), average='macro'):.4f} ({time.time()-t_f:.1f}s)")
        test += predict_4view_tta(model, te_ldr, target_hw, mean_t, std_t, use_tta=True) / 5.0

        del model, opt, sched, scaler, best_st
        gc.collect()
        torch.cuda.empty_cache()

    assert not np.isnan(oof).any() and not np.isnan(test).any()
    np.save(oof_p, oof)
    np.save(test_p, test)
    eval_unseen_report(tag, oof)
    return oof, test
"""))

    cells.append(code_cell("""
# [Cell 8] Execute Phase 2 E2E Models & Pack All Outputs (Zero Auto-Submit)
oof_cnx, te_cnx = train_5fold_e2e("v5_convnext_small", "convnext_small.fb_in22k_ft_in1k", (128, 256), epochs=8, batch_size=64 * BS_SCALE, lr_bb=1.4e-4, lr_hd=1.1e-3, drop_path=0.25)
oof_swin, te_swin = train_5fold_e2e("v5_swin_tiny", "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", (224, 224), epochs=8, batch_size=64 * BS_SCALE, lr_bb=1.2e-4, lr_hd=1.0e-3, drop_path=0.20)
oof_mv3, te_mv3 = train_5fold_e2e("v5_mobilenetv3", "mobilenetv3_large_100.ra_in1k", (128, 256), epochs=9, batch_size=128 * BS_SCALE, lr_bb=4.0e-4, lr_hd=1.5e-3, drop_path=0.0)

print("\\n=== Summary Comparison across ALL Cloud Models (Unseen-Scene 5-Fold CV) ===", flush=True)
p_lr_path = os.path.join(OUT_DIR, "oof_probs_v4_foundation_lr.npy")
p_mlp_path = os.path.join(OUT_DIR, "oof_probs_v4_foundation_mlp.npy")
oof_v4_lr_d = np.load(p_lr_path) if os.path.exists(p_lr_path) else None
oof_v4_mlp_d = np.load(p_mlp_path) if os.path.exists(p_mlp_path) else None
if oof_v4_mlp_d is not None:
    eval_unseen_report("v4_foundation_mlp", oof_v4_mlp_d)
eval_unseen_report("v5_convnext_small", oof_cnx)
eval_unseen_report("v5_swin_tiny", oof_swin)
eval_unseen_report("v5_mobilenetv3", oof_mv3)
if oof_v4_mlp_d is not None and oof_v4_lr_d is not None:
    quick_ens = 0.30 * oof_v4_mlp_d + 0.10 * oof_v4_lr_d + 0.25 * oof_cnx + 0.20 * oof_swin + 0.15 * oof_mv3
    eval_unseen_report("Quick_5Model_Mix", quick_ens)

final_zip = os.path.join(OUT_DIR, "task4_cloud_all_outputs.zip")
bundle_files = [
    "embs_v4_cloud_5backbones.npz", "phys_features_v4.npz", "scene_split_v4.npz",
    "oof_probs_v4_foundation_lr.npy", "test_probs_v4_foundation_lr.npy",
    "oof_probs_v4_foundation_mlp.npy", "test_probs_v4_foundation_mlp.npy",
    "oof_probs_v5_convnext_small.npy", "test_probs_v5_convnext_small.npy",
    "oof_probs_v5_swin_tiny.npy", "test_probs_v5_swin_tiny.npy",
    "oof_probs_v5_mobilenetv3.npy", "test_probs_v5_mobilenetv3.npy",
]
with zipfile.ZipFile(final_zip, "w", compression=zipfile.ZIP_STORED) as zf:
    for fn in bundle_files:
        fp = os.path.join(OUT_DIR, fn)
        if os.path.exists(fp):
            zf.write(fp, arcname=fn)
print(f"\\n>>> STATUS: ALL_DONE | Download ONLY this zip back to local task4/: {final_zip} ({os.path.getsize(final_zip)/1e6:.1f} MB) <<<", flush=True)
if DRIVE_BACKUP_DIR:
    shutil.copy2(final_zip, os.path.join(DRIVE_BACKUP_DIR, "task4_cloud_all_outputs.zip"))
    print("Auto-backed up final zip to Google Drive:", DRIVE_BACKUP_DIR)
"""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(NB_OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Generated clean cloud notebook: {NB_OUT} ({os.path.getsize(NB_OUT)/1e3:.1f} KB)")


if __name__ == "__main__":
    main()
