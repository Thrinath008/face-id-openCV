import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

os.makedirs("gaze_data", exist_ok=True)
DATA_PATH = "gaze_data/gaze_data.npz"


def extract_eye_features(landmarks):
    """
    Extract 4D gaze features from eye corners + eyelids + iris centers.

    MediaPipe face mesh indices (approx):
      Left eye:
        - outer corner: 33
        - inner corner: 133
        - upper eyelid: 159
        - lower eyelid: 145
        - iris center: 468
      Right eye:
        - outer corner: 263
        - inner corner: 362
        - upper eyelid: 386
        - lower eyelid: 374
        - iris center: 473
    """

    def xy(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y], dtype=np.float32)

    # Left eye
    L_outer = xy(33)
    L_inner = xy(133)
    L_up = xy(159)
    L_down = xy(145)
    L_iris = xy(468)

    # Right eye
    R_outer = xy(263)
    R_inner = xy(362)
    R_up = xy(386)
    R_down = xy(374)
    R_iris = xy(473)

    # Horizontal ratios
    L_w = L_inner[0] - L_outer[0]
    R_w = R_inner[0] - R_outer[0]

    if abs(L_w) < 1e-6:
        L_h = 0.5
    else:
        L_h = (L_iris[0] - L_outer[0]) / L_w

    if abs(R_w) < 1e-6:
        R_h = 0.5
    else:
        R_h = (R_iris[0] - R_outer[0]) / R_w

    # Vertical ratios (0 = upper lid, 1 = lower lid)
    L_hh = L_down[1] - L_up[1]
    R_hh = R_down[1] - R_up[1]

    if abs(L_hh) < 1e-6:
        L_v = 0.5
    else:
        L_v = (L_iris[1] - L_up[1]) / L_hh

    if abs(R_hh) < 1e-6:
        R_v = 0.5
    else:
        R_v = (R_iris[1] - R_up[1]) / R_hh

    # Clip to a sane range
    L_h = float(np.clip(L_h, -0.5, 1.5))
    R_h = float(np.clip(R_h, -0.5, 1.5))
    L_v = float(np.clip(L_v, -0.5, 1.5))
    R_v = float(np.clip(R_v, -0.5, 1.5))

    return np.array([L_h, L_v, R_h, R_v], dtype=np.float32)


def main():
    cap = cv2.VideoCapture(0)

    print("Gaze data collection")
    print("Press '1' when you are LOOKING AT THE SCREEN (positive class)")
    print("Press '0' when you are LOOKING AWAY (negative class)")
    print("Press 'q' to quit and save.")

    X = []
    y = []

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                )

                features = extract_eye_features(face_landmarks.landmark)

                cv2.putText(
                    frame,
                    "Press 1: screen | 0: away | q: save",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Gaze Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("1"):
                    X.append(features)
                    y.append(1)
                    print(f"Captured sample #{len(X)} (label=1: screen)")
                elif key == ord("0"):
                    X.append(features)
                    y.append(0)
                    print(f"Captured sample #{len(X)} (label=0: away)")
                elif key == ord("q"):
                    X = np.array(X)
                    y = np.array(y)
                    np.savez(DATA_PATH, X=X, y=y)
                    print(f"Saved {len(X)} samples to {DATA_PATH}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            else:
                cv2.imshow("Gaze Data Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()