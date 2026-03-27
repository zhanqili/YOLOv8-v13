import os, sys, time, re, io
from contextlib import redirect_stdout

import cv2
import numpy as np
import torch
import torch.serialization as ts

# ====== 关键：优先使用当前 yolov9 仓库 ======
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ====== PyTorch 2.6+ 权重加载兼容（信任的 best.pt）======
ts.add_safe_globals([np.core.multiarray._reconstruct])
_torch_load = torch.load
def torch_load_unsafe(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = torch_load_unsafe
# ============================================================

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression

# =========  =========
WEIGHTS = r"runs/train/exp/weights/best.pt"  # 改成你的 yolov9 best.pt
IMGSZ = 640
DEVICE = "cuda:0"
WARMUP = 30
RUNS = 200
CONF_THRES = 0.25
IOU_THRES = 0.45
MAX_DET = 300
TEST_IMAGE = None  # e.g. r"data/images/bus.jpg"
# ============================


def file_mb(p):
    return os.path.getsize(p) / (1024 * 1024)


def preprocess(img0, imgsz, device):
    img, _, _ = letterbox(img0, new_shape=imgsz, auto=False)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img)
    x = torch.from_numpy(img).to(device).float() / 255.0
    return x.unsqueeze(0)


@torch.no_grad()
def e2e_once(model, img0, device):
    x = preprocess(img0, IMGSZ, device)
    pred = model(x)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    _ = non_max_suppression(pred, CONF_THRES, IOU_THRES, max_det=MAX_DET)


def parse_summary(text: str):
    """
    从 yolov9 打印的 summary 里抓 params 和 GFLOPs
    例：gelan summary: 372 layers, 2846951 parameters, 0 gradients, 11.3 GFLOPs
    """
    m_params = re.search(r"([\d,]+)\s+parameters", text)
    m_gflops = re.search(r"([0-9]*\.?[0-9]+)\s+GFLOPs", text)
    params = int(m_params.group(1).replace(",", "")) if m_params else None
    gflops = float(m_gflops.group(1)) if m_gflops else None
    return params, gflops


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("repo root:", ROOT)

    weights = WEIGHTS
    if not os.path.isabs(weights):
        weights = os.path.join(ROOT, weights)

    # 读取模型，同时捕获控制台输出（含 “... parameters ... GFLOPs”）
    buf = io.StringIO()
    with redirect_stdout(buf):
        dmb = DetectMultiBackend(weights, device=device)
    log = buf.getvalue()

    params, gflops = parse_summary(log)


    m = dmb.model
    m.eval()
    if params is None:
        params = sum(p.numel() for p in m.parameters())
    params_m = params / 1e6

    # 图片输入
    if TEST_IMAGE:
        img0 = cv2.imread(TEST_IMAGE)
        if img0 is None:
            raise FileNotFoundError(TEST_IMAGE)
    else:
        img0 = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)

    # warmup
    for _ in range(WARMUP):
        e2e_once(dmb, img0, device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # bench
    t0 = time.perf_counter()
    for _ in range(RUNS):
        e2e_once(dmb, img0, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    lat_ms = (t1 - t0) * 1000 / RUNS
    fps = 1000 / lat_ms

    print("\n===== YOLOv9 End-to-End Metrics =====")
    print("weights:", weights)
    print(f"Model Size (MB): {file_mb(weights):.2f}")
    print(f"Params (M): {params_m:.3f}")
    print(f"FLOPs (G @ {IMGSZ}): {gflops if gflops is not None else 'N/A'}  (from repo summary)")
    print(f"Latency (ms, e2e): {lat_ms:.3f}")
    print(f"FPS (e2e): {fps:.2f}")


    if log.strip():
        print("\n[Captured summary log]")
        print(log.strip())


if __name__ == "__main__":
    main()