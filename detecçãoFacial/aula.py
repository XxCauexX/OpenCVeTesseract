import pytesseract
import mediapipe as mp # pip install mediapipe
import cv2

# inicializar o opencv e o mediapipe
webcam = cv2.VideoCapture(0)
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

while True:
    # Ler as informações da webcamo
    verificador, frame = webcam.read()

    if not verificador:
        break

    # Reconhecer os rostos que tem ali dentro
    lista_rostos = reconhecedor_rostos.process(frame)

    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            #desenhar os rostos na imagem
            desenho.draw_detection(frame, rosto)
    cv2.imshow("Rostos na webcam", frame)

    # quando apertat ESC, para o loop
    if cv2.waitKey(5) == 27:
        break

webcam.release()
cv2.destroyAllWindows()