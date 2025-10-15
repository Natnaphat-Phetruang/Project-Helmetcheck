from __future__ import annotations
import argparse, json, time
from datetime import datetime
from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np

# ==== optional: YOLO (helmet) ====
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ==== mediapipe ====
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh
except Exception:
    mp = None
    mp_face = None

# ---------------- paths & const ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ID_DIR       = PROJECT_ROOT / "identify" / "id_export"
LBPH_MODEL   = ID_DIR / "lbph_model.yml"
LABELS_JSON  = ID_DIR / "labels.json"
BEST_PT      = PROJECT_ROOT / "model_best" / "best.pt"

HEAD_PORTION     = 0.48
CARD_W_RATIO     = 0.40
DISPLAY_DURATION = 8.0
HELMET_HOLD_SEC  = 8.0
PREDICT_ANGLES   = [-15,-10,-6,-3,0,3,6,10,15]

# ---------------- utils ----------------
def s2b(s: str) -> bool:
    return str(s).lower() in ("1","true","yes","on","y")

def draw_corners(img, x1, y1, x2, y2, color=(0,215,255), lw=3, l=24):
    cv2.line(img,(x1,y1),(x1+l,y1),color,lw); cv2.line(img,(x1,y1),(x1,y1+l),color,lw)
    cv2.line(img,(x2,y1),(x2-l,y1),color,lw); cv2.line(img,(x2,y1),(x2,y1+l),color,lw)
    cv2.line(img,(x1,y2),(x1+l,y2),color,lw); cv2.line(img,(x1,y2),(x1,y2-l),color,lw)
    cv2.line(img,(x2,y2),(x2-l,y2),color,lw); cv2.line(img,(x2,y2),(x2,y2-l),color,lw)

def draw_card(img, name, in_db, helmet_text):
    h, w = img.shape[:2]
    x0, y0 = 12, h - 118
    card_w = int(w * CARD_W_RATIO)
    cv2.rectangle(img, (x0, y0), (x0 + card_w, y0 + 106), (40,40,40), -1)
    cv2.rectangle(img, (x0, y0), (x0 + card_w, y0 + 34), (105,105,105), -1)
    cv2.putText(img, "TRACKING", (x0+10, y0+24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, name, (x0+12, y0+60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    in_txt   = "IN DATABASE" if (in_db and name!="Unknown") else "NOT IN DATABASE"
    in_color = (110,255,110) if (in_db and name!="Unknown") else (60,200,255)
    cv2.putText(img, in_txt, (x0+12, y0+82), cv2.FONT_HERSHEY_SIMPLEX, 0.78, in_color, 2, cv2.LINE_AA)
    cv2.putText(img, f"HELMET: {helmet_text}", (x0+12, y0+104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (110,255,110) if helmet_text=="HELMET" else ((60,200,255) if helmet_text=="NO HELMET" else (200,200,200)),
                2, cv2.LINE_AA)

def preprocess_gray(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160,160))
    gray = cv2.equalizeHist(gray)
    return gray

def make_box(face_box, mode="face"):
    x1,y1,x2,y2 = face_box
    if mode=="head":
        h = y2-y1
        head_h = int(h*HEAD_PORTION)
        hy1 = max(0, y1 - int(head_h*0.15))
        hy2 = y1 + head_h
        return (x1,hy1,x2,hy2)
    return face_box

def now_stamp():
    dt = datetime.now()
    return dt.strftime("%Y%m%d-%H%M%S"), f"{int(dt.microsecond/1000):03d}"

# ---- quality & alignment (ใช้ใน still-verify) ----
def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def face_quality_score(bgr, box):
    x1,y1,x2,y2 = box
    x1=max(0,x1); y1=max(0,y1); x2=min(bgr.shape[1]-1,x2); y2=min(bgr.shape[0]-1,y2)
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return -1.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sharp = laplacian_var(gray)
    area = (x2-x1)*(y2-y1) / float(bgr.shape[0]*bgr.shape[1] + 1e-6)
    ratio = (y2-y1) / float((x2-x1) + 1e-6)
    frontal = 1.0 - abs(ratio - 1.2)
    return 0.6*min(sharp/1500.0, 1.0) + 0.3*min(area*5, 1.0) + 0.1*max(min(frontal,1),0)

def align_by_eyes(bgr, output_size=(160,160)):
    if mp_face is None:
        return None, False
    h, w = bgr.shape[:2]
    with mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None, False
        lm = res.multi_face_landmarks[0].landmark
        def pt(i): return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)
        right = 0.5*(pt(33)+pt(133))
        left  = 0.5*(pt(362)+pt(263))
        d = left - right
        angle = np.degrees(np.arctan2(d[1], d[0]))
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rot = cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
        return cv2.resize(cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY), output_size), True

# ---------------- detectors ----------------
class FaceDetector:
    def __init__(self, method="haar"):
        self.method = method
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.dnn = None
        if method=="dnn":
            mdl = Path(__file__).with_suffix("").parent / "dnn_models"
            proto = mdl / "deploy.prototxt"
            caffemodel = mdl / "res10_300x300_ssd_iter_140000.caffemodel"
            if proto.exists() and caffemodel.exists():
                self.dnn = cv2.dnn.readNetFromCaffe(str(proto), str(caffemodel))
            else:
                print("[WARN] DNN model not found, fallback to HAAR.")
                self.method = "haar"

    def detect(self, frame):
        H, W = frame.shape[:2]
        if self.method=="dnn" and self.dnn is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300,300), (104,177,123))
            self.dnn.setInput(blob)
            det = self.dnn.forward()
            faces=[]
            for i in range(det.shape[2]):
                conf = det[0,0,i,2]
                if conf < 0.5: continue
                x1,y1,x2,y2 = (det[0,0,i,3:7]*np.array([W,H,W,H])).astype(int)
                faces.append((max(0,x1),max(0,y1),min(W-1,x2),min(H-1,y2)))
            return faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.haar.detectMultiScale(gray, 1.2, 5, minSize=(70,70))
        return [(x,y,x+w,y+h) for (x,y,w,h) in boxes]

def predict_lbph_best(recog, roi_gray):
    h,w = roi_gray.shape
    best = (None, float("inf"))
    for a in PREDICT_ANGLES:
        test = roi_gray if a==0 else cv2.warpAffine(
            roi_gray, cv2.getRotationMatrix2D((w//2,h//2), a, 1.0), (w,h),
            borderMode=cv2.BORDER_REFLECT101
        )
        label, dist = recog.predict(test)
        if dist < best[1]:
            best = (label, dist)
    return best

def predict_lbph_strict(recog, roi_gray, angles=tuple(range(-20,21,2))):
    h,w = roi_gray.shape
    best = (None, float("inf"))
    for a in angles:
        if a == 0:
            test = roi_gray
        else:
            M = cv2.getRotationMatrix2D((w//2,h//2), a, 1.0)
            test = cv2.warpAffine(roi_gray, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
        label, dist = recog.predict(test)
        if dist < best[1]:
            best = (label, dist)
    return best

# ---------------- main ----------------
def run(weights: Path,
        source="0",
        face_detector="haar",
        lbph_thr: float = 120.0,
        name_smooth: int = 10,
        enable_id: bool = True,
        enable_helmet: bool = True,
        box_mode: str = "face",
        window_size: str = "1280x720",
        process_every: int = 1,
        helmet_conf: float = 0.60,
        helmet_confirm: int = 3,
        target_fps: int = 60,
        yolo_every: int = 5,
        imgsz: int = 480,
        cap_width: int = 640,
        cap_height: int = 360,
        cv2_threads: int = -1,
        combo_hold: float = 5.0,
        verify_k: int = 8,
        verify_dist_thr: float = 100.0,
        verify_outdir: Path = PROJECT_ROOT / "logs" / "verify_queue",
        ):

    # OpenCV optimizations
    cv2.setUseOptimized(True)
    if cv2_threads >= 0:
        cv2.setNumThreads(cv2_threads)

    # Logs / dirs
    logdir = PROJECT_ROOT / "logs"
    logdir.mkdir(exist_ok=True)
    screens_dir = logdir / "screens"; screens_dir.mkdir(exist_ok=True)
    verify_dir = Path(verify_outdir); verify_dir.mkdir(parents=True, exist_ok=True)

    # ID model
    recog, labels = None, {}
    if enable_id:
        assert LBPH_MODEL.exists() and LABELS_JSON.exists(), "Run identify/build_face_db_single.py ก่อน"
        recog = cv2.face.LBPHFaceRecognizer_create()
        recog.read(str(LBPH_MODEL))
        labels = {int(k): v for k,v in json.loads(LABELS_JSON.read_text('utf-8')).items()}

    # Helmet model
    helmet = None
    if enable_helmet and YOLO is not None:
        if not Path(weights).exists(): raise FileNotFoundError(f"Helmet weights not found: {weights}")
        helmet = YOLO(str(weights))

    # Video
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except: pass

    # UI resize
    resize_target = None
    if "x" in window_size:
        try: resize_target = tuple(map(int, window_size.lower().split("x")))
        except: pass

    # States
    alpha = 0.6
    prev_boxes = []
    name_buf = deque(maxlen=max(3, int(name_smooth)))
    last_name, last_in_db, t_name = "Unknown", False, 0.0
    last_helmet, t_helmet = "UNKNOWN", 0.0
    fd = FaceDetector(method=face_detector)

    helmet_buf = deque(maxlen=max(3, helmet_confirm))
    cached_helmet = "UNKNOWN"

    # combo hold state
    combo_lock_until = 0.0
    combo_lock_name  = "Unknown"
    combo_lock_helmet= "UNKNOWN"

    # keep recent samples for still-verify
    recent_samples = deque(maxlen=60)  # (frame, box, timestamp)

    frame_idx = 0
    while True:
        loop_start = time.time()
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        # ---------- 1) Face detect + ID ----------
        current_name, current_in_db = "Unknown", False
        current_label, current_dist = -1, None
        boxes=[]
        do_work = (frame_idx % max(1,process_every) == 0)

        if do_work:
            faces = fd.detect(frame)
            people=[]
            for (x1,y1,x2,y2) in faces:
                head = make_box((x1,y1,x2,y2), mode=box_mode)
                name, in_db = "Unknown", False
                pred_label, pred_dist = -1, None
                if enable_id and recog is not None:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size!=0:
                        roi = preprocess_gray(crop)
                        pred_label, pred_dist = predict_lbph_best(recog, roi)
                        if pred_dist <= lbph_thr:
                            name = labels.get(int(pred_label), "Unknown")
                            in_db = (name!="Unknown")
                people.append({"box": head, "name": name, "in_db": in_db, "label": pred_label, "dist": pred_dist})

            # smooth box
            cur_boxes = [p["box"] for p in people]
            if prev_boxes and len(prev_boxes)==len(cur_boxes):
                sm=[]
                for (x1,y1,x2,y2),(px1,py1,px2,py2) in zip(cur_boxes, prev_boxes):
                    sm.append((int(alpha*px1+(1-alpha)*x1), int(alpha*py1+(1-alpha)*y1),
                               int(alpha*px2+(1-alpha)*x2), int(alpha*py2+(1-alpha)*y2)))
                for p,b in zip(people, sm): p["box"]=b
                prev_boxes = sm
            else:
                prev_boxes = cur_boxes[:]

            if people:
                bi = max(range(len(people)), key=lambda i:(people[i]["box"][2]-people[i]["box"][0])*(people[i]["box"][3]-people[i]["box"][1]))
                raw = people[bi]
                boxes=[raw["box"]]
                name_buf.append(raw["name"])
                most,cnt = Counter(name_buf).most_common(1)[0]
                current_name = most if (most!="Unknown" and cnt>=name_buf.maxlen) else "Unknown"
                current_in_db = (current_name!="Unknown")
                current_label, current_dist = raw["label"], raw["dist"]

        if current_name=="Unknown" and (time.time()-t_name)<DISPLAY_DURATION and last_name!="Unknown":
            current_name,current_in_db = last_name,last_in_db
        elif current_name!="Unknown" and current_name!=last_name:
            last_name,last_in_db,t_name = current_name,current_in_db,time.time()

        # still-verify
        if boxes:
            x1,y1,x2,y2 = boxes[0]
            recent_samples.append((frame.copy(), (x1,y1,x2,y2), time.time()))

        # ---------- 2) Helmet detect ----------
        ran_yolo = False
        helmet_text = "UNKNOWN"
        if enable_helmet and helmet is not None:
            if (frame_idx % max(1, yolo_every)) == 0:
                ran_yolo = True
                res = helmet.predict(frame, imgsz=imgsz, conf=helmet_conf, verbose=False)
                pred = "NO HELMET"
                face_box = boxes[0] if boxes else None

                def _is_valid_helmet_box(bxyxy, face_box, frame_h):
                    x1,y1,x2,y2=bxyxy; bw, bh=(x2-x1),(y2-y1)
                    if bh<=0 or bw<=0: return False
                    if y1>frame_h*0.40 or bh>frame_h*0.55: return False
                    if face_box is None: return True
                    fx1,fy1,fx2,fy2=face_box; fh=max(1,fy2-fy1)
                    forehead_y=fy1+int(0.30*fh)
                    overlap_y1,overlap_y2=max(y1,fy1),min(y2,forehead_y)
                    if overlap_y2-overlap_y1<=0: return False
                    if y2>fy1+int(0.55*fh): return False
                    return True

                for b in res[0].boxes:
                    c = int(b.cls.item()); conf = float(b.conf.item())
                    cls_name = res[0].names[c].lower() if hasattr(res[0],"names") else str(c)
                    if "helmet" in cls_name and conf >= helmet_conf:
                        x1,y1,x2,y2 = map(int, b.xyxy[0])
                        if _is_valid_helmet_box((x1,y1,x2,y2), face_box, frame.shape[0]):
                            pred = "HELMET"; break

                helmet_buf.append(pred)
                if len(helmet_buf)==helmet_buf.maxlen:
                    helmet_text = "HELMET" if Counter(helmet_buf).most_common(1)[0][0]=="HELMET" else "NO HELMET"
                else:
                    helmet_text = "UNKNOWN"
            else:
                # (cached majority buffer)
                if len(helmet_buf):
                    helmet_text = "HELMET" if Counter(helmet_buf).most_common(1)[0][0]=="HELMET" else "NO HELMET"

        if helmet_text=="UNKNOWN" and (time.time()-t_helmet)<HELMET_HOLD_SEC:
            helmet_text=last_helmet
        elif helmet_text!="UNKNOWN":
            last_helmet, t_helmet = helmet_text, time.time()

        # ---------- 2.5) (name+helmet) ----------
        now = time.time()
        both_green = (current_name != "Unknown" and current_in_db and helmet_text == "HELMET")
        if both_green:
            combo_lock_until = now + float(combo_hold)
            combo_lock_name   = current_name
            combo_lock_helmet = "HELMET"
        if now < combo_lock_until:
            current_name  = combo_lock_name
            current_in_db = True
            helmet_text   = combo_lock_helmet

        # ---------- 3) Draw ----------
        canvas=frame.copy()
        if boxes:
            x1,y1,x2,y2=boxes[0]
            draw_corners(canvas,x1,y1,x2,y2,(0,215,255),4,26)
        draw_card(canvas,current_name,current_in_db,helmet_text)

        if resize_target: canvas=cv2.resize(canvas,resize_target)
        cv2.imshow(f"Helmet + ID (q/Esc quit) | thr={lbph_thr:.1f}",canvas)

        # --------- frame limiter ----------
        elapsed = time.time() - loop_start
        if target_fps > 0:
            min_frame_time = 1.0 / float(target_fps)
            delay_ms = int(max(1, (min_frame_time - elapsed) * 1000)) if elapsed < min_frame_time else 1
        else:
            delay_ms = 1

        key=cv2.waitKey(delay_ms)&0xFF
        if key in (27,ord('q')): break
        elif key==ord(']'): lbph_thr+=2.0
        elif key==ord('['): lbph_thr=max(10.0,lbph_thr-2.0)
        elif key==ord('s'):
            ts,ms=now_stamp(); shot=(logdir/"screens"/f"{ts}_{ms}_frame{frame_idx:06d}.png")
            cv2.imwrite(str(shot),canvas); print(f"[SCREEN] saved: {shot}")
        # ---- Still-verify: capture best frames ----
        elif key==ord('c'):
            if not recent_samples:
                print("[VERIFY] no recent samples to capture.")
            else:
                ts, ms = now_stamp()
                batch_dir = verify_dir / f"batch_{ts}_{ms}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                scored = []
                for fr, bx, _ in list(recent_samples):
                    scored.append((face_quality_score(fr, bx), fr, bx))
                scored.sort(key=lambda x: x[0], reverse=True)
                saved = 0
                take = max(5, int(name_smooth))
                for i,(sc,fr,bx) in enumerate(scored[:take]):
                    x1,y1,x2,y2 = bx
                    crop = fr[y1:y2, x1:x2]
                    cv2.imwrite(str(batch_dir/f"sample_{i:02d}.png"), crop)
                    saved += 1
                print(f"[VERIFY] saved {saved} samples to {batch_dir}")
        elif key==ord('v'):
            batches = sorted(verify_dir.glob("batch_*"))
            if not batches:
                print("[VERIFY] no batch folder, press 'c' to capture first.")
            else:
                batch = batches[-1]
                items = sorted(batch.glob("*.png"))
                if not items:
                    print("[VERIFY] empty batch:", batch)
                else:
                    results = []
                    for p in items:
                        bgr = cv2.imread(str(p))
                        if bgr is None: continue
                        aligned, ok = align_by_eyes(bgr, output_size=(160,160))
                        if ok and aligned is not None:
                            roi = cv2.equalizeHist(aligned)
                        else:
                            roi = preprocess_gray(bgr)
                        label, dist = predict_lbph_strict(recog, roi)
                        results.append((p.name, label, dist))
                    if not results:
                        print("[VERIFY] no valid images to verify.")
                    else:
                        results.sort(key=lambda x: x[2])
                        topk = results[:max(1, int(verify_k))]
                        labels_cnt = Counter([l for _,l,_ in topk if l is not None])
                        if labels_cnt:
                            best_label, count = labels_cnt.most_common(1)[0]
                            best_name = labels.get(int(best_label), "Unknown")
                            mean_dist = float(np.mean([d for _,l,d in topk if l==best_label]))
                            is_in = (mean_dist <= float(verify_dist_thr)) and (best_name!="Unknown")
                            print(f"[VERIFY] result: name={best_name}  mean_dist={mean_dist:.1f}  vote={count}/{len(topk)}  IN={is_in}")
                        else:
                            print("[VERIFY] result: Unknown")

    cap.release(); cv2.destroyAllWindows()

# ---------------- cli ----------------
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="0")
    ap.add_argument("--weights",type=Path,default=(PROJECT_ROOT/"model_best"/"best.pt"))
    ap.add_argument("--face-detector",choices=["haar","dnn"],default="haar")
    ap.add_argument("--lbph-thr",type=float,default=120.0)
    ap.add_argument("--name-smooth",type=int,default=10)
    ap.add_argument("--enable-id",type=str,default="true")
    ap.add_argument("--enable-helmet",type=str,default="true")
    ap.add_argument("--box-mode",choices=["face","head"],default="face")
    ap.add_argument("--window-size",type=str,default="1280x720")
    ap.add_argument("--process-every",type=int,default=1)
    ap.add_argument("--helmet-conf",type=float,default=0.60)
    ap.add_argument("--helmet-confirm",type=int,default=3)
    ap.add_argument("--target-fps",type=int,default=60)
    ap.add_argument("--yolo-every",type=int,default=5)
    ap.add_argument("--imgsz",type=int,default=480)
    ap.add_argument("--cap-width",type=int,default=640)
    ap.add_argument("--cap-height",type=int,default=360)
    ap.add_argument("--cv2-threads",type=int,default=-1)
    ap.add_argument("--combo-hold", type=float, default=5.0)
    ap.add_argument("--verify-k", type=int, default=8)
    ap.add_argument("--verify-dist-thr", type=float, default=100.0)
    ap.add_argument("--verify-outdir", type=Path, default=(PROJECT_ROOT/"logs"/"verify_queue"))
    return ap.parse_args()

if __name__=="__main__":
    args=parse_args()
    run(weights=args.weights,source=args.source,face_detector=args.face_detector,
        lbph_thr=args.lbph_thr,name_smooth=args.name_smooth,
        enable_id=s2b(args.enable_id),enable_helmet=s2b(args.enable_helmet),
        box_mode=args.box_mode,window_size=args.window_size,
        process_every=max(1,int(args.process_every)),
        helmet_conf=args.helmet_conf,helmet_confirm=max(1,int(args.helmet_confirm)),
        target_fps=args.target_fps,yolo_every=max(1,int(args.yolo_every)),
        imgsz=args.imgsz,cap_width=args.cap_width,cap_height=args.cap_height,
        cv2_threads=args.cv2_threads,
        combo_hold=args.combo_hold,
        verify_k=args.verify_k,
        verify_dist_thr=args.verify_dist_thr,
        verify_outdir=args.verify_outdir)
