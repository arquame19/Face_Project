import cv2
import face_recognition
import numpy as np
import time
import mediapipe as mp
import os
import datetime
import pickle
from ultralytics import YOLO
import logging


#  LOGGING SETUP
if not os.path.exists("logs"):
    os.makedirs("logs")

log_filename = f"logs/proctoring_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("Proctoring System Started")

#  LOAD YOUR TRAINED MODEL
#  (trained on proctoring_dataset.csv)
model  = pickle.load(open("ml/model.pkl",  "rb"))
scaler = pickle.load(open("ml/scaler.pkl", "rb"))
logging.info(f"ML Model loaded — expects {scaler.n_features_in_} features")

#  SETUP
if not os.path.exists("cheating"):
    os.makedirs("cheating")

yolo_model = YOLO("yolov8n.pt")

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(max_num_faces=2)

#  LOAD FACE DATASET
known_encodings = []
known_names     = []
dataset_path    = "dataset"

logging.info("Loading dataset...")
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        try:
            image     = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                logging.info(f"Loaded dataset image: {person_name}/{img_name}")
        except:
            logging.error(f"Error loading image: {img_path}")
logging.info("Dataset Loaded Successfully")

# ──────────────────────────────────────────
#  CAMERA
# ──────────────────────────────────────────
cap = cv2.VideoCapture(0)

prev_time         = 0
frame_count       = 0
gaze_status       = "Center"
recognized_name   = "Unknown"
unknown_count     = 0
last_capture_time = 0
score_history     = []

# Head pose constants (solvePnP)
FACE_3D = np.array([
    [0.0,    0.0,    0.0],
    [0.0,  -330.0,  -65.0],
    [-225.0, 170.0, -135.0],
    [225.0,  170.0, -135.0],
    [-150.0,-150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)
POSE_IDS   = [1, 152, 263, 33, 287, 57]
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]

# ──────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────
def get_head_pose(landmarks, w, h):
    pts = np.array([[landmarks[i].x*w, landmarks[i].y*h]
                    for i in POSE_IDS], dtype=np.float64)
    focal = w
    cam   = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(FACE_3D, pts, cam, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles[1]*360, angles[0]*360   # yaw, pitch

def get_ear(landmarks, idx, w, h):
    p = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in idx]
    A = np.linalg.norm(np.array(p[1])-np.array(p[5]))
    B = np.linalg.norm(np.array(p[2])-np.array(p[4]))
    C = np.linalg.norm(np.array(p[0])-np.array(p[3]))
    return (A+B)/(2.0*C+1e-6)

def save_cheating_frame(frame, reason):
    global last_capture_time
    if time.time() - last_capture_time > 3:
        filename = f"cheating/{reason}_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        logging.warning(f"Cheating detected: {reason} | Saved: {filename}")
        last_capture_time = time.time()

def recognize_face(frame, x1, y1, x2, y2):
    global recognized_name, unknown_count
    try:
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (150, 150))
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if encs:
            dists = face_recognition.face_distance(known_encodings, encs[0])
            best  = np.argmin(dists)
            if dists[best] < 0.5:
                recognized_name = known_names[best]
                unknown_count   = 0
            else:
                recognized_name = "Unknown"
                unknown_count  += 1
                logging.warning("Unknown face detected")
                save_cheating_frame(frame, "unknown_face")
                if unknown_count >= 3:
                    cap.release()
                    cv2.destroyAllWindows()
                    logging.critical("Unauthorized face detected — system exiting")
                    exit()
    except:
        pass

# ──────────────────────────────────────────
#  ML PREDICT — exactly 10 features
#  matching proctoring_dataset.csv columns:
#
#  1.  no_person
#  2.  multiple_persons
#  3.  phone_detected
#  4.  gaze_off_screen
#  5.  eyes_closed
#  6.  head_yaw_norm
#  7.  head_pitch_norm
#  8.  ear_norm
#  9.  face_absent
#  10. unknown_face
# ──────────────────────────────────────────
def get_suspicion_score(person_count, phone_detected,
                        gaze_off, eyes_closed,
                        head_yaw, head_pitch,
                        ear_avg, face_absent,
                        recognized_name):

    features = [
        float(person_count == 0),               # 1. no_person
        float(person_count > 1),                # 2. multiple_persons
        float(phone_detected),                  # 3. phone_detected
        float(gaze_off),                        # 4. gaze_off_screen
        float(eyes_closed),                     # 5. eyes_closed
        min(abs(head_yaw)   / 90.0, 1.0),      # 6. head_yaw_norm
        min(abs(head_pitch) / 90.0, 1.0),      # 7. head_pitch_norm
        max(0.0, 1.0 - ear_avg * 3.0),         # 8. ear_norm
        float(face_absent),                     # 9. face_absent
        float(recognized_name == "Unknown"),    # 10. unknown_face
    ]

    arr    = np.array(features, dtype=np.float32).reshape(1, -1)
    scaled = scaler.transform(arr)
    return float(model.predict_proba(scaled)[0][1])

# ──────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    small = cv2.resize(frame, (320, 240))
    h, w  = frame.shape[:2]
    frame_count += 1

    person_count   = 0
    phone_detected = False
    head_yaw       = 0.0
    head_pitch     = 0.0
    ear_avg        = 0.30
    gaze_off       = False
    eyes_closed    = False
    face_absent    = False

    # ── YOLO every 3 frames ──────────────────────
    if frame_count % 3 == 0:
        results = yolo_model(small, verbose=False)
        for r in results:
            for box in r.boxes:
                cls          = int(box.cls[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                x1*=2; y1*=2; x2*=2; y2*=2

                if cls == 0:
                    person_count += 1
                    if frame_count % 10 == 0:
                        recognize_face(frame, x1, y1, x2, y2)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame, recognized_name,(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                if cls == 67:
                    phone_detected = True
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(frame,"PHONE DETECTED",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    # ── MediaPipe every 4 frames ──────────────────
    if frame_count % 4 == 0:
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            sh, sw = 240, 320

            # Gaze (original logic)
            lx = int(lm[33].x  * sw) * 2
            rx = int(lm[263].x * sw) * 2
            nx = int(lm[1].x   * sw) * 2
            center = (lx + rx) // 2
            if nx < center - 20:
                gaze_status = "Left"
            elif nx > center + 20:
                gaze_status = "Right"
            else:
                gaze_status = "Center"

            # Head pose
            head_yaw, head_pitch = get_head_pose(lm, sw*2, sh*2)
            gaze_off = abs(head_yaw) > 20 or abs(head_pitch) > 20

            # Eye Aspect Ratio
            l_ear   = get_ear(lm, LEFT_EYE,  sw*2, sh*2)
            r_ear   = get_ear(lm, RIGHT_EYE, sw*2, sh*2)
            ear_avg = (l_ear + r_ear) / 2
            eyes_closed = l_ear < 0.20 and r_ear < 0.20
        else:
            gaze_status = "No face"
            face_absent = True

    # ── ML Score ──────────────────────────────────
    score = get_suspicion_score(
        person_count, phone_detected,
        gaze_off, eyes_closed,
        head_yaw, head_pitch,
        ear_avg, face_absent,
        recognized_name
    )

    score_history.append(score)
    if len(score_history) > 15:
        score_history.pop(0)
    smooth_score = float(np.mean(score_history))

    # ── Cheating Conditions ────────────────────────
    if phone_detected:
        save_cheating_frame(frame, "phone")
        logging.warning("Phone detected")

    if person_count > 1:
        cv2.putText(frame,"MULTIPLE PERSONS",(50,100),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        save_cheating_frame(frame, "multiple")
        logging.warning(f"Multiple persons detected: {person_count}")

    if gaze_status != "Center":
        save_cheating_frame(frame, "gaze")
        logging.info(f"Gaze deviation: {gaze_status}")

    if smooth_score >= 0.7:
        save_cheating_frame(frame, "ml_suspicious")
        logging.info(f"Suspicion Score: {smooth_score:.2f}")
    
    if face_absent:
        logging.warning("No face detected in frame")

    # ── FPS ───────────────────────────────────────
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # ── UI ────────────────────────────────────────
    cv2.putText(frame, f"FPS: {int(fps)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(frame, f"Gaze: {gaze_status}", (200,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,
                (0,255,0) if gaze_status=="Center" else (0,0,255),2)

    # Suspicion bar
    bar_w = int(w * smooth_score)
    color = (0,200,0) if smooth_score < 0.4 else \
            (0,165,255) if smooth_score < 0.7 else (0,0,255)
    cv2.rectangle(frame,(0,h-22),(w,h),    (30,30,30),-1)
    cv2.rectangle(frame,(0,h-22),(bar_w,h), color,    -1)
    cv2.putText(frame, f"Suspicion: {smooth_score:.2f}",
                (5,h-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    cv2.imshow("AI PROCTORING SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
logging.info("Proctoring System Stopped")
cap.release()
cv2.destroyAllWindows()