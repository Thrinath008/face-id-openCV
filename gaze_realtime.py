import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

MODEL_PATH = "gaze_data/gaze_model.pkl"


def extract_eye_features(landmarks):
    """
    Same 4D feature extractor as collect_gaze_data.py
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

    L_h = float(np.clip(L_h, -0.5, 1.5))
    R_h = float(np.clip(R_h, -0.5, 1.5))
    L_v = float(np.clip(L_v, -0.5, 1.5))
    R_v = float(np.clip(R_v, -0.5, 1.5))

    return np.array([L_h, L_v, R_h, R_v], dtype=np.float32)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run train_gaze_model.py first.")

    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting real-time gaze detection. Press 'q' to quit.")

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

            label_text = "No face detected"
            color = (0, 0, 255)

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
                prob = clf.predict_proba([features])[0][1]  # P(class=1: screen)

                if prob > 0.5:
                    label_text = f"Looking at screen ({prob:.2f})"
                    color = (0, 255, 0)
                else:
                    label_text = f"Looking away ({prob:.2f})"
                    color = (0, 0, 255)

            cv2.putText(
                frame,
                label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            cv2.imshow("Gaze Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()