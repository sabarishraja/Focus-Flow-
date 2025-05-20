import cv2
import mediapipe as mp

#Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces = 1)

#Drawing utility to draw face mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#Webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignore empty frame")
        continue
    #Flip the image horizontally for mirror view
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Process the frame of video with Face Mesh
    result = face_mesh.process(rgb_image)

    #Draw Facial Landmarks
    if result.multi_face_landmarks:  #This will show a list of landmark sets on the face
        for face_landmarks in result.multi_face_landmarks:
            lm = face_landmarks.landmark[468]
            print(f"Left pupil -> x: {lm.x:.3f}, y: {lm.y:.3f}")
            break
            # mp_drawing.draw_landmarks(
            #     image = image,
            #     landmark_list = face_landmarks,
            #     connections = mp_face_mesh.FACEMESH_TESSELATION, 
            #     landmark_drawing_spec = None, 
            #     connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
            # )
    cv2.imshow('Mediapipe Face Mesh', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()