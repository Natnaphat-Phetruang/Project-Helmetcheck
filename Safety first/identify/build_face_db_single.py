# identify/build_face_db_single.py
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

# ============ PATHS ============
ROOT = Path(__file__).resolve().parents[1]
KNOWN = ROOT / "identify" / "known_faces"
OUT   = ROOT / "identify" / "id_export"
OUT.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_HW = (160, 160)

# ============ FACE DETECTORS ============
class FaceDetector:
    """Select between HAAR and DNN (Res10-SSD)."""
    def __init__(self, method: str = "haar",
                 scale_factor: float = 1.2,
                 min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (60, 60)):
        self.method = method.lower().strip()
        self.scale = scale_factor
        self.nei   = min_neighbors
        self.min_size = tuple(min_size)

        self.haar = None
        self.dnn  = None

        if self.method == "dnn":
            mdl_dir = ROOT / "identify" / "dnn_models"
            proto = mdl_dir / "deploy.prototxt"
            caff = mdl_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            if proto.exists() and caff.exists():
                self.dnn = cv2.dnn.readNetFromCaffe(str(proto), str(caff))
            else:
                print("[WARN] DNN model files not found in identify/dnn_models/, fallback to HAAR.")
                self.method = "haar"

        if self.method == "haar":
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        H, W = bgr.shape[:2]
        if self.method == "dnn" and self.dnn is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)),
                                         1.0, (300, 300), (104, 177, 123))
            self.dnn.setInput(blob)
            det = self.dnn.forward()
            faces = []
            for i in range(det.shape[2]):
                conf = float(det[0, 0, i, 2])
                if conf < 0.5:
                    continue
                x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([W, H, W, H])).astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2, y2))
            return faces

        # HAAR
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.haar.detectMultiScale(
            gray, self.scale, self.nei, minSize=self.min_size
        )
        return [(x, y, x + w, y + h) for (x, y, w, h) in boxes]

# ============ UTILS ============
def sha1_hex(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def now_stamp() -> Tuple[str, str]:
    dt = datetime.now()
    return dt.strftime("%Y%m%d-%H%M%S"), f"{int(dt.microsecond/1000):03d}"

def safe_slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")

def detect_face_crop(bgr: np.ndarray, det: FaceDetector) -> Tuple[np.ndarray, bool]:
    """Return (roi_gray_160x160, found). If not found -> resize full gray."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = det.detect(bgr)
    if not faces:
        return cv2.resize(gray, FACE_HW), False
    x1, y1, x2, y2 = max(faces, key=lambda r: (r[2]-r[0])*(r[3]-r[1]))
    roi = gray[y1:y2, x1:x2]
    return cv2.resize(roi, FACE_HW), True

def augment_gray(img: np.ndarray) -> List[np.ndarray]:
    """Light augmentations for small datasets."""
    def rot(a):
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), a, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
    def clahe(x):
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return c.apply(x)
    def bc(alpha=1.15, beta=0):
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    def blur(k=3):
        return cv2.GaussianBlur(img, (k, k), 0)

    xs = [img, rot(+8), rot(-8), cv2.flip(img, 1),
          bc(1.15, +10), bc(0.85, -10), blur(3), clahe(img)]
    n = np.random.normal(0, 5, img.shape).astype(np.float32)
    xs.append(np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8))
    h, w = img.shape
    xs.append(cv2.resize(img[4:h-4, 4:w-4], (w, h)))
    xs.append(cv2.resize(img[0:h-6, 0:w-6], (w, h)))
    xs.append(cv2.resize(img[3:h, 3:w], (w, h)))
    return xs[:12]

def list_images(d: Path) -> Iterable[Path]:
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            yield p

def bucket_of(path: Path) -> str:
    name = path.parent.name.lower()
    if name == "no_helmet": return "no"
    if name == "wear_helmet": return "wear"
    return "na"

def read_display_name(person_dir: Path) -> str:
    dn = person_dir / "display_name.txt"
    if dn.exists():
        try:
            t = dn.read_text(encoding="utf-8").strip()
            if t:
                return t
        except:
            pass
    return person_dir.name

# ============ LOGGING ============
def ensure_csv_header(csv_path: Path):
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "ts", "person_id", "person_name", "bucket",
                "found_face", "seq",
                "crop_path", "src_path", "src_sha1",
                "orig_basename", "detector"
            ])

def write_logs(csv_path: Path, jsonl_path: Path, row: dict):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            row["ts"], row["person_id"], row["person_name"], row["bucket"],
            int(row["found_face"]), row["seq"],
            row["crop_path"], row["src_path"], row["src_sha1"],
            row["orig_basename"], row["detector"]
        ])
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ============ DATA COLLECTION ============
def collect_person_files(person_dir: Path, mode: str, balance: bool,
                         max_per_person: int | None, rng: np.random.Generator) -> Tuple[List[Path], int, int]:
    nh_dir = person_dir / "no_helmet"
    wh_dir = person_dir / "wear_helmet"
    nh_list = list(list_images(nh_dir)) if nh_dir.exists() else []
    wh_list = list(list_images(wh_dir)) if wh_dir.exists() else []

    use_nh = (mode in ("both", "no_helmet")) and nh_list
    use_wh = (mode in ("both", "wear_helmet")) and wh_list

    if not use_nh and not use_wh:
        files = list(list_images(person_dir))
        rng.shuffle(files)
        if max_per_person:
            files = files[:max_per_person]
        return files, len(files), 0

    rng.shuffle(nh_list); rng.shuffle(wh_list)
    if not use_nh: nh_list = []
    if not use_wh: wh_list = []
    if balance and nh_list and wh_list:
        m = min(len(nh_list), len(wh_list))
        nh_list = nh_list[:m]; wh_list = wh_list[:m]

    files = nh_list + wh_list
    rng.shuffle(files)
    if max_per_person and len(files) > max_per_person:
        files = files[:max_per_person]
    return files, len(nh_list), len(wh_list)

# ============ MAIN ============
def main():
    ap = argparse.ArgumentParser(description="Build LBPH face DB from known_faces; crop & log to logs/")
    ap.add_argument("--mode", choices=["both","no_helmet","wear_helmet"], default="both",
                    help="เลือกแหล่งรูปต่อคน (default: both)")
    ap.add_argument("--balance", action="store_true",
                    help="บาลานซ์จำนวนรูป no_helmet / wear_helmet ให้เท่ากันถ้ามีทั้งสอง")
    ap.add_argument("--max-per-person", type=int, default=None,
                    help="จำนวนรูปสูงสุดต่อคน (หลัง balance/shuffle)")
    ap.add_argument("--seed", type=int, default=12345, help="random seed")

    # Logging / crops
    ap.add_argument("--logdir", type=str, default=str(ROOT / "logs"),
                    help="โฟลเดอร์สำหรับ logs/crops (default: ./logs)")
    ap.add_argument("--no-save-crops", action="store_true", help="ไม่เซฟไฟล์ครอป (default คือเซฟ)")

    # Face detector params
    ap.add_argument("--face-detector", choices=["haar","dnn"], default="haar",
                    help="ตัวตรวจจับใบหน้า (haar = เร็ว, dnn = แม่นกว่า)")
    ap.add_argument("--haar-scale", type=float, default=1.2, help="HAAR scaleFactor")
    ap.add_argument("--haar-neighbors", type=int, default=5, help="HAAR minNeighbors")
    ap.add_argument("--haar-minsize", type=str, default="60x60", help="HAAR minSize, เช่น 60x60")

    # LBPH params
    ap.add_argument("--lbph-radius", type=int, default=2)
    ap.add_argument("--lbph-neighbors", type=int, default=8)
    ap.add_argument("--lbph-gridx", type=int, default=8)
    ap.add_argument("--lbph-gridy", type=int, default=8)

    args = ap.parse_args()

    # parse HAAR minSize
    try:
        mw, mh = map(int, args.haar_minsize.lower().split("x"))
        haar_minsize = (mw, mh)
    except:
        haar_minsize = (60, 60)

    # prepare logs
    LOG_DIR = Path(args.logdir)
    CROP_ROOT = LOG_DIR / "crops"
    CROP_ROOT.mkdir(parents=True, exist_ok=True)
    CSV_PATH = LOG_DIR / "build_faces.csv"
    JSONL_PATH = LOG_DIR / "build_faces.jsonl"
    ensure_csv_header(CSV_PATH)

    rng = np.random.default_rng(args.seed)

    persons = [d for d in sorted(KNOWN.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    if not persons:
        raise RuntimeError(f"Put person folders & images under: {KNOWN}")

    # Init detector
    det = FaceDetector(
        method=args.face_detector,
        scale_factor=args.haar_scale,
        min_neighbors=args.haar_neighbors,
        min_size=haar_minsize
    )

    images, labels = [], []
    id2name = {}
    cur_id = 0
    total_found_face = 0
    total_images = 0

    for pdir in persons:
        files, nh_cnt, wh_cnt = collect_person_files(
            pdir, mode=args.mode, balance=args.balance,
            max_per_person=args.max_per_person, rng=rng
        )
        if not files:
            print(f"[WARN] no images in {pdir}, skip")
            continue

        person_name = read_display_name(pdir)
        id2name[cur_id] = person_name
        pslug = f"pid{cur_id:03d}_{safe_slug(person_name)}"
        crop_dir = CROP_ROOT / pslug
        crop_dir.mkdir(parents=True, exist_ok=True)

        rois = []
        found_count = 0
        seq = 0

        for f in files:
            bgr = cv2.imread(str(f))
            if bgr is None:
                print(f"[WARN] cannot read {f}, skip")
                continue
            roi, found = detect_face_crop(bgr, det)
            roi = cv2.equalizeHist(roi)
            rois.append(roi)
            found_count += int(found)

            if not args.no_save_crops:
                seq += 1
                ts, ms = now_stamp()
                bkt = bucket_of(f)
                src_sha = sha1_hex(f)
                base = f.stem
                fname = (
                    f"{ts}_{ms}__{pslug}__bucket-{bkt}"
                    f"__found-{int(found)}__seq{seq:04d}"
                    f"__src-{safe_slug(base)}__sha{src_sha[:8]}.png"
                )
                crop_p = crop_dir / fname
                cv2.imwrite(str(crop_p), roi)

                row = {
                    "ts": f"{ts}.{ms}",
                    "person_id": cur_id,
                    "person_name": person_name,
                    "bucket": bkt,
                    "found_face": bool(found),
                    "seq": seq,
                    "crop_path": str(crop_p.relative_to(ROOT)),
                    "src_path": str(f.relative_to(ROOT) if ROOT in f.parents else str(f)),
                    "src_sha1": src_sha,
                    "orig_basename": f.name,
                    "detector": det.method,
                }
                write_logs(CSV_PATH, JSONL_PATH, row)

        if len(rois) == 1:
            rois = augment_gray(rois[0])

        for r in rois:
            images.append(r)
            labels.append(cur_id)

        total_images += len(files)
        total_found_face += found_count
        print(f"[OK] {person_name}: {len(rois)} samples  "
              f"(picked: {len(files)} | no_helmet={nh_cnt}, wear_helmet={wh_cnt}, "
              f"face_found={found_count}, crops_saved={len(files) if not args.no_save_crops else 0})")
        cur_id += 1

    if not images:
        raise RuntimeError("No training samples.")

    # Train LBPH
    recog = cv2.face.LBPHFaceRecognizer_create(
        radius=args.lbph_radius,
        neighbors=args.lbph_neighbors,
        grid_x=args.lbph_gridx,
        grid_y=args.lbph_gridy
    )
    recog.train(images, np.array(labels, dtype=np.int32))

    model_p  = OUT / "lbph_model.yml"
    labels_p = OUT / "labels.json"
    recog.write(str(model_p))
    labels_p.write_text(json.dumps({str(i): n for i, n in id2name.items()},
                                   ensure_ascii=False, indent=2), "utf-8")

    print("\n=== SUMMARY ===")
    print(f"persons: {len(id2name)}")
    print(f"picked images: {total_images} | faces detected: {total_found_face}  (detector: {det.method})")
    print(f"[SAVED] {model_p}")
    print(f"[SAVED] {labels_p}")
    print(f"[LOG]   {CSV_PATH}")
    print(f"[LOG]   {JSONL_PATH}")
    if not args.no_save_crops:
        print(f"[CROPS] {CROP_ROOT}")

if __name__ == "__main__":
    main()
