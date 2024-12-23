import cv2
import mediapipe as mp

# Inicializar MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

KEYPOINTS = [1, 33, 263, 61, 291, 199]  # Nariz, ojos y extremos de la boca

with mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # Convertir a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for id in KEYPOINTS:
                    landmark = face_landmarks.landmark[id]
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  
        cv2.imshow('Key Points Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
