import cv2
import mediapipe as mp
import socket
import json

# Inicializar MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

# Configuración de red
HOST = '127.0.0.1'  # IP local
PORT = 5006         # Puerto que Unity usará
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Puntos clave a enviar (nariz, ojos, extremos de la boca)
KEYPOINTS = [1, 33, 263, 61, 291]  # 5 puntos clave

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
                keypoints_data = {}
                for id in KEYPOINTS:
                    landmark = face_landmarks.landmark[id]
                    keypoints_data[id] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    }
                # Enviar datos a Unity
                sock.sendto(json.dumps(keypoints_data).encode(), (HOST, PORT))
                print("Enviando datos:", json.dumps(keypoints_data))
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
            break

cap.release()
sock.close()
cv2.destroyAllWindows()
