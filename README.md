คู่มือการใช้งานระบบตรวจจับหมวกนิรภัย (แบบ 2 ขั้นตอน)

ระบบนี้ทำงานเป็น 2 ส่วนหลัก
1. เทรนโมเดล (Train Model) บน Google Colab เพราะต้องใช้ GPU แรง
2. รันระบบจริง (Run Detection) บนเครื่องตัวเองผ่าน VS Code เพื่อใช้กล้องตรวจจับ

------------------------------------------------------------
ขั้นตอนที่ 1 : เทรนโมเดล บน Google Colab ( Code ส่วนนี้อยู่ใน Folder Train Model )
------------------------------------------------------------

1. เตรียม Kaggle API Key
- เข้า https://www.kaggle.com
- ล็อกอิน > คลิกไอคอนโปรไฟล์ (มุมขวาบน) > เลือก Account / Settings
- เลื่อนลงมาที่หัวข้อ API
- คลิก Create New API Token
- จะได้ไฟล์ชื่อ kaggle.json (เก็บไว้ในเครื่อง)

2. เข้า Colab และ Run
- เปิดไฟล์ helmet_detector_demo.ipynb ใน Google Colab กด Run all
- ระบบจะ Run จากบนลงล่าง
- เมื่อ Run มาถึง  Cell 1 — Kaggle API (Drive fallback)
- จะมี pop-up ขอใช้ใน google drive ให้อนุญาต
- ตรงส่วนด้านล่างของ Cell จะมีให้ใส่ไฟล์ kaggle.json เพื่อใช้ในการ Download Dataset
- (Data Set) https://www.kaggle.com/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3

3. เทรนโมเดล YOLO
- เมื่อ Run มาถึง  Cell 5 — Train YOLOv8
- จะทำการ Train เป็นจำนวน 20 epoch (จะเหลือ 1 epoch ในตัว Demo)

4. ดาวน์โหลดไฟล์โมเดล best.pt
- เมื่อ Run มาถึง  Cell 8 — Download Model
- จะทำการ Download file Weight (best.pt) อัตโนมัติให้ทำการเก็บไฟล์เอาไว้ที่ Folder model_best

------------------------------------------------------------
ขั้นตอนที่ 2 : นำโมเดลไปใช้ บน VS Code
------------------------------------------------------------

1. วางไฟล์โมเดล
- นำไฟล์ best.pt ที่ได้จาก Colab มาวางไว้ในโฟลเดอร์:
  project_root/model_best/

  ตัวอย่างโครงสร้างโฟลเดอร์:
  project_root/
  ├─ helmet_detect/
  │  └─ helmet_app_v3_lbph.py
  ├─ model_best/
  │  └─ best.pt
  ├─ identify/
  │  ├─ known_faces/
  │  └─ id_export/
  └─ logs/

2. ติดตั้ง Library ที่จำเป็น
-เปิด folder > helmet_detect
-ดูด้านซ้าย คลิกขวาที่ helmet_app_v3_lbph.py แล้ว เลือก open in integranted terminal
-ดูที่ terminal ด้านล่าง จะต้องอยู่ใน Path .....\.....\helmet_detect>

พิมพ์คำสั่ง:
pip install "opencv-contrib-python>=4.8" numpy ultralytics mediapipe pandas pillow tqdm


3. ติดตั้ง Library ที่จำเป็น
-เปิด folder > known_faces
-ดูด้านซ้าย คลิกขวาที่ build_face_db_single.py แล้ว เลือก open in integranted terminal
-ดูที่ terminal ด้านล่าง จะต้องอยู่ใน Path .....\.....\identify\known_faces>

พิมพ์คำสั่ง:
pip install "opencv-contrib-python>=4.8" numpy ultralytics mediapipe pandas pillow tqdm


4. รันโปรแกรมตรวจจับหมวก
- เปิด VS Code
- ไปที่โฟลเดอร์ known_faces
- สร้าง Folder สำหรับเก็บภาพพนักงาน โดยจะ
   ├─ display_name.txt        # ใส่ “ชื่อ–นามสกุล”
   └─ no_helmet/              # (แนะนำ) ใส่รูปหน้าตรง/เอียงซ้าย 45°/เอียงขวา 45° อย่างละ 1 รูปขึ้นไป

- คลิกขวาไฟล์ build_face_db_single.py > Open in Integrated Terminal
- พิมพ์คำสั่ง:
  python build_face_db_single.py
- เพื่อทำการสร้าง ฐานข้อมูลของพนักงาน


5. รันโปรแกรมตรวจจับหมวก
- เปิด VS Code
- ไปที่โฟลเดอร์ helmet_detect
- คลิกขวาไฟล์ helmet_app_v3_lbph.py > Open in Integrated Terminal
- พิมพ์คำสั่ง:
  python helmet_app_v3_lbph.py


6. โปรแกรมจะเปิดกล้องขึ้นมา
- แสดงกรอบหน้าและชื่อพนักงาน (ถ้ามีในฐานข้อมูล)
- แสดงสถานะหมวก: HELMET / NO HELMET / UNKNOWN

7. ปิดโปรแกรม
- กด Q หรือ ESC เพื่อออกจากโปรแกรม

8. ตรวจสอบผลลัพธ์
- ภาพที่ถูกจับจะอยู่ในโฟลเดอร์:
  project_root/logs/screens/
- ภาพที่เก็บไว้ตรวจภายหลังจะอยู่ที่:
  project_root/logs/verify_queue/
