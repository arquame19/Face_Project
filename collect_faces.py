import cv2
import os

name = input("Enter person name: ")
path = f"dataset/{name}"

os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    if not ret:
        break

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        img_path = f"{path}/{count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()