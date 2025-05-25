import cv2
import mediapipe as mp
import numpy as np

# ——— Setup MediaPipe Face Mesh ———
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ——— Landmark indices ———
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

LEFT_EYE_LIDS = [159, 145]
RIGHT_EYE_LIDS = [386, 374]

LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# ——— Open webcam ———
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            def draw_eye(iris_idxs, corner_idxs, lid_idxs, outline_idxs, label):
                # ——— Eye Outline using polylines ———
                outline_points = np.array([
                    (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
                    for idx in outline_idxs
                ])
                cv2.polylines(frame, [outline_points], isClosed=True, color=(0, 255, 255), thickness=1)  # Electric Blue

                # ——— Iris landmarks (larger dots) ———
                for idx in iris_idxs:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (255, 105, 180), -1)  # Hot Pink, Bigger Dots

                # ——— Eye corners ———
                corner_coords = []
                for idx in corner_idxs:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    corner_coords.append((x, y))
                    cv2.circle(frame, (x, y), 2, (50, 255, 50), -1)  # Lime

                # ——— Eyelids ———
                y_top = face_landmarks.landmark[lid_idxs[0]].y * h
                y_bottom = face_landmarks.landmark[lid_idxs[1]].y * h
                for idx in lid_idxs:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 165, 255), -1)  # Orange

                # ——— Eye center ———
                if len(corner_coords) == 2:
                    x_center = (corner_coords[0][0] + corner_coords[1][0]) // 2
                    y_center = int((y_top + y_bottom) / 2)
                    cv2.circle(frame, (x_center, y_center), 2, (0, 0, 255), -1)  # Red
                    cv2.putText(frame, f'{label} Eye Center', (x_center + 5, y_center - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)  # Red

            # ——— Draw both eyes ———
            draw_eye(LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, LEFT_EYE_OUTLINE, "Left")
            draw_eye(RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, RIGHT_EYE_OUTLINE, "Right")

    cv2.imshow('Both Eye Tracking (Vibrant Pupils)', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
