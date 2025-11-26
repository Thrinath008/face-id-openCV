import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import torch

# Select device: use Apple GPU (MPS) if available, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

def open_camera(max_index=3):
    """Try camera indices from 0..max_index-1 and return the first that opens."""
    for idx in range(max_index):
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            print(f"Using camera index: {idx}")
            cam.set(cv2.CAP_PROP_FPS, 60)  # target 60fps preview if supported
            return cam
        cam.release()
    raise RuntimeError("No camera device found. Check connections and try again.")

# Run MTCNN on CPU to avoid MPS adaptive pool limitations
detector = MTCNN(keep_all=False, device='cpu')
model = InceptionResnetV1(pretrained='vggface2').to(device).eval()

my_emb = np.load("embeddings/me_embedding.npy")

def is_me(emb):
    dist = np.linalg.norm(emb - my_emb)
    return dist < 0.9   # you can tune this threshold

cam = open_camera()
# Reduce resolution a bit for higher FPS
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Starting real-time verification... Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    # Convert to RGB once per frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = detector.detect(rgb)

    if boxes is not None:
        x1, y1, x2, y2 = boxes[0].astype(int)
        face = rgb[y1:y2, x1:x2]

        if face.size > 0:
            face_resized = cv2.resize(face, (160,160))
            face_tensor = torch.tensor(face_resized).permute(2,0,1).float().unsqueeze(0) / 255
            face_tensor = face_tensor.to(device)

            with torch.no_grad():
                emb = model(face_tensor).detach().cpu().numpy()[0]

            label = "THRINATH" if is_me(emb) else "Unknown❌"
            color = (0,255,0) if label == "THRINATH" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Verification", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()