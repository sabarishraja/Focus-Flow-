import cv2
import mediapipe as mp
import numpy as np
import math

# ——— Setup MediaPipe Face Mesh ———
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ——— Landmark indices ———
LEFT_IRIS        = [468, 469, 470, 471, 472]
RIGHT_IRIS       = [473, 474, 475, 476, 477]

LEFT_EYE_CORNERS  = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

LEFT_EYE_LIDS     = [159, 145]
RIGHT_EYE_LIDS    = [386, 374]

LEFT_EYE_OUTLINE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# ——— Angular thresholds (in degrees) ———
# Tweak these to your liking
CENTER_ANGLE   = 30    # ±30° → center
LEFT_ANGLE_MIN = 135   # beyond ±135° → left
LEFT_ANGLE_MAX = 180
RIGHT_ANGLE_MIN = -180 # or +180 
RIGHT_ANGLE_MAX = -135
UP_ANGLE_MIN    = -135
UP_ANGLE_MAX    = -45
DOWN_ANGLE_MIN  = 45
DOWN_ANGLE_MAX  = 135

# ——— Open webcam & get FPS ———
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.release()

# ——— Main loop ———
cap = cv2.VideoCapture(0)
off_center_count = 0
ALERT_SECONDS = 2.0
ALERT_FRAMES  = int(ALERT_SECONDS * fps)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results = face_mesh.process(rgb_frame)
    directions = []  # collect each eye’s “direction”

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            def draw_eye_angular(iris_idxs, corner_idxs, lid_idxs, outline_idxs, label):
                # ——— Outline ———
                pts = np.array([
                    (int(face_landmarks.landmark[i].x * w),
                     int(face_landmarks.landmark[i].y * h))
                    for i in outline_idxs
                ])
                cv2.polylines(frame, [pts], True, (0, 255, 255), 1)

                # ——— Corners ———
                corners = []
                for i in corner_idxs:
                    x = int(face_landmarks.landmark[i].x * w)
                    y = int(face_landmarks.landmark[i].y * h)
                    corners.append((x, y))
                    cv2.circle(frame, (x, y), 2, (50, 255, 50), -1)

                # ——— Eyelids ———
                y_top    = face_landmarks.landmark[lid_idxs[0]].y * h
                y_bottom = face_landmarks.landmark[lid_idxs[1]].y * h
                for i in lid_idxs:
                    x = int(face_landmarks.landmark[i].x * w)
                    y = int(face_landmarks.landmark[i].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 165, 255), -1)

                # ——— Iris & center ———
                iris_pts = []
                for i in iris_idxs:
                    x = int(face_landmarks.landmark[i].x * w)
                    y = int(face_landmarks.landmark[i].y * h)
                    iris_pts.append((x, y))
                    cv2.circle(frame, (x, y), 2, (255, 105, 180), -1)

                (cx, cy), _ = cv2.minEnclosingCircle(np.array(iris_pts, dtype=np.int32))
                cx, cy = int(cx), int(cy)
                cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

                # ——— Compute eye center ———
                # Horizontal midpoint between the two corner landmarks
                x_center = (corners[0][0] + corners[1][0]) // 2
                # Vertical midpoint between top and bottom lid
                y_center = int((y_top + y_bottom) / 2)

                # draw eye-center if you want:
                cv2.circle(frame, (x_center, y_center), 2, (0, 0, 255), -1)

                # ——— Compute angle (in degrees) ———
                dx = cx - x_center
                dy = cy - y_center
                theta = math.degrees(math.atan2(dy, dx))
                # θ runs from -180° to +180°:
                #  0° → purely to the right,  ±90° → down/up,  ±180° → left

                # ——— Classify by angle ———
                if -CENTER_ANGLE <= theta <= +CENTER_ANGLE:
                    d = 'Center'
                elif (theta >= LEFT_ANGLE_MIN and theta <= LEFT_ANGLE_MAX) or \
                     (theta <= RIGHT_ANGLE_MAX and theta <= RIGHT_ANGLE_MIN):
                    # angles near ±180° → left
                    d = 'Left'
                elif UP_ANGLE_MIN <= theta <= UP_ANGLE_MAX:
                    d = 'Up'
                elif DOWN_ANGLE_MIN <= theta <= DOWN_ANGLE_MAX:
                    d = 'Down'
                else:
                    # anything between +30°→+150° is “Down-Right” etc. 
                    # But for simplicity, if > +CENTER_ANGLE and < +DOWN_ANGLE_MIN, label “Right”
                    if theta > CENTER_ANGLE and theta < DOWN_ANGLE_MIN:
                        d = 'Right'
                    else:
                        d = 'Left'  # fallback

                directions.append(d)
                # draw a tiny text label “L” or “R” near iris center
                cv2.putText(frame, label, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            # Draw & classify both eyes
            draw_eye_angular(LEFT_IRIS,  LEFT_EYE_CORNERS,  LEFT_EYE_LIDS,  LEFT_EYE_OUTLINE,  'L')
            draw_eye_angular(RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, RIGHT_EYE_OUTLINE, 'R')

    # ——— Determine overall gaze (majority vote) ———
    gaze = None
    if directions:
        counts = {d: directions.count(d) for d in set(directions)}
        gaze = max(counts, key=counts.get)
        cv2.putText(frame, f'Gaze: {gaze}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    # ——— Distraction Alert Logic ———
    if gaze and gaze != 'Center':
        off_center_count += 1
    else:
        off_center_count = 0

    if off_center_count > ALERT_FRAMES:
        # semi-transparent red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, 'FOCUS!', (w // 3, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)

    # ——— Show frame ———
    cv2.imshow('Angular Gaze + Distraction Alert', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
