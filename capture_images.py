import cv2
from facenet_pytorch import MTCNN
import os
from PIL import Image

os.makedirs("data/train_me", exist_ok=True)

detector = MTCNN(keep_all=False)

cam = cv2.VideoCapture(0)
count = 0

print("Capturing your face images...")

while count < 40:
    ret, frame = cam.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face
    boxes, _ = detector.detect(rgb)

    if boxes is not None:
        x1, y1, x2, y2 = boxes[0].astype(int)
        face = rgb[y1:y2, x1:x2]

        if face.size > 0:
            img = Image.fromarray(face)
            img.save(f"data/train_me/{count}.jpg")
            count += 1
            print(f"Captured {count}/40")

    cv2.imshow("Capturing...", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()