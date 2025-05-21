import cv2
import mediapipe as mp

# ——— Setup Mediapipe Face Mesh ———
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# ——— Define only the eye‐region connections ———
LEFT_EYE_CONNECTIONS  = set(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_CONNECTIONS = set(mp_face_mesh.FACEMESH_RIGHT_EYE)
EYE_CONNECTIONS       = LEFT_EYE_CONNECTIONS | RIGHT_EYE_CONNECTIONS

# ——— Precompute the unique eye‐landmark indices ———
eye_indices = { idx for connection in EYE_CONNECTIONS for idx in connection }

# ——— Open webcam ———
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Mirror & convert to RGB
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face Mesh processing
    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            # Draw only the eye‐region connections with red lines
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=EYE_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
            )

            # Draw small red circles at each eye landmark
            h, w, _ = image.shape
            for idx in eye_indices:
                lm = face_landmarks.landmark[idx]
                x_px, y_px = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x_px, y_px), 1, (0,0,255), -1)  # radius=2, filled

    cv2.imshow('Eye Mesh Only (Red)', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
