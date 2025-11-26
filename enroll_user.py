import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import torch
from PIL import Image
import os
import argparse

# ---------- args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="Username to enroll")
parser.add_argument("--cam", type=int, default=1, help="Camera index")
parser.add_argument("--shots", type=int, default=30, help="Number of face samples")
args = parser.parse_args()

username = args.name.strip()
assert username, "Name cannot be empty"

os.makedirs("embeddings", exist_ok=True)

detector = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

cam = cv2.VideoCapture(args.cam)
cam.set(cv2.CAP_PROP_FPS, 60)

print(f"Enrolling user: {username}")
print("Look at the camera. Press 'q' to abort.")

embs = []

while len(embs) < args.shots:
    ret, frame = cam.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = detector.detect(rgb)

    if boxes is not None:
        x1, y1, x2, y2 = boxes[0].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(0, x2), max(0, y2)
        face = rgb[y1:y2, x1:x2]

        if face.size > 0:
            face_resized = cv2.resize(face, (160, 160))
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float().unsqueeze(0) / 255

            with torch.no_grad():
                emb = model(face_tensor).numpy()[0]
                embs.append(emb)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing {len(embs)}/{args.shots}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Enroll User", frame)
    if cv2.waitKey(1) == ord('q'):
        print("Enrollment aborted.")
        cam.release()
        cv2.destroyAllWindows()
        exit()

cam.release()
cv2.destroyAllWindows()

embs = np.array(embs)
mean_emb = embs.mean(axis=0)
out_path = os.path.join("embeddings", f"{username}.npy")
np.save(out_path, mean_emb)

print(f"Enrollment complete. Saved embedding to {out_path}")