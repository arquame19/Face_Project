import cv2
import face_recognition
import numpy as np
import time
import mediapipe as mp
import os
import datetime
from ultralytics import YOLO

# 🔥 Create cheating folder
if not os.path.exists("cheating"):
    os.makedirs("cheating")

# 🔥 Load YOLO model
model = YOLO("yolov8n.pt")

# 🔹 Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)

# 🔥 LOAD DATASET (instead of encodings.pkl)
known_encodings = []
known_names = []

dataset_path = "dataset"

print("Loading dataset...")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        try:
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                print(f"Loaded: {person_name} - {img_name}")
        except:
            print(f"Error loading {img_path}")

print("Dataset Loaded ✅")

# 🔹 Camera
cap = cv2.VideoCapture(0)

prev_time = 0
frame_count = 0
gaze_status = "Center"

recognized_name = "Unknown"
unknown_count = 0
last_capture_time = 0

# 🔥 Save cheating frame
def save_cheating_frame(frame, reason):
    global last_capture_time
    if time.time() - last_capture_time > 3:
        filename = f"cheating/{reason}_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print("Captured:", filename)
        last_capture_time = time.time()

# 🔥 FACE RECOGNITION FUNCTION
def recognize_face(frame, x1, y1, x2, y2):
    global recognized_name, unknown_count

    face_crop = frame[y1:y2, x1:x2]

    try:
        face_crop = cv2.resize(face_crop, (150, 150))
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        if encodings:
            face_encoding = encodings[0]

            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                    recognized_name = known_names[best_match_index]
                    unknown_count = 0
                else:
                    recognized_name = "Unknown"
                    unknown_count += 1

                    print("❌ Unknown face detected!")

                    save_cheating_frame(frame, "unknown_face")

                    # 🔥 EXIT after 3 detections
                    if unknown_count >= 3:
                        print("🚫 Exiting due to unauthorized face")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
    except:
        pass

# 🔁 MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    small = cv2.resize(frame, (320, 240))

    person_count = 0
    phone_detected = False

    # 🔥 YOLO detection (every 3 frames)
    if frame_count % 3 == 0:
        results = model(small, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Scale coordinates
                x1 *= 2; y1 *= 2; x2 *= 2; y2 *= 2

                # 👤 PERSON
                if cls == 0:
                    person_count += 1

                    if frame_count % 10 == 0:
                        recognize_face(frame, x1, y1, x2, y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, recognized_name,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

                # 📱 PHONE
                if cls == 67:
                    phone_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, "PHONE DETECTED",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

    # 🔹 Gaze detection
    if frame_count % 4 == 0:
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for lm in result.multi_face_landmarks:
                h, w, _ = small.shape

                left = lm.landmark[33]
                right = lm.landmark[263]
                nose = lm.landmark[1]

                lx = int(left.x * w) * 2
                rx = int(right.x * w) * 2
                nx = int(nose.x * w) * 2

                center = (lx + rx) // 2

                if nx < center - 20:
                    gaze_status = "Left"
                elif nx > center + 20:
                    gaze_status = "Right"
                else:
                    gaze_status = "Center"

    # 🔥 CHEATING CONDITIONS

    if phone_detected:
        save_cheating_frame(frame, "phone")

    if person_count > 1:
        cv2.putText(frame, "MULTIPLE PERSONS",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        save_cheating_frame(frame, "multiple")

    if gaze_status != "Center":
        save_cheating_frame(frame, "gaze")

    # 🔹 FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # 🔹 UI
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Gaze: {gaze_status}", (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if gaze_status == "Center" else (0, 0, 255), 2)

    cv2.imshow("AI PROCTORING SYSTEM", frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()