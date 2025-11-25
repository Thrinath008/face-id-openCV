import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import torch

detector = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

my_emb = np.load("embeddings/me_embedding.npy")

def is_me(emb):
    dist = np.linalg.norm(emb - my_emb)
    return dist < 0.9   # you can tune this threshold

cam = cv2.VideoCapture(0)
print("Starting real-time verification... Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = detector.detect(rgb)

    if boxes is not None:
        x1, y1, x2, y2 = boxes[0].astype(int)
        face = rgb[y1:y2, x1:x2]

        if face.size > 0:
            face_resized = cv2.resize(face, (160,160))
            face_tensor = torch.tensor(face_resized).permute(2,0,1).float().unsqueeze(0) / 255

            with torch.no_grad():
                emb = model(face_tensor).numpy()[0]

            label = "THRINATH" if is_me(emb) else "Unknown❌"
            color = (0,255,0) if label == "THRINATH" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Verification", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()