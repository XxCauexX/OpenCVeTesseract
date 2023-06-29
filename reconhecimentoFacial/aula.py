import cv2 # biblioteca do openCV
import numpy as np

print(cv2.__version__)
#Detecção de faces com Haar cascade (OpenCV)
imagem = cv2.imread("face/person.png")
print(imagem.shape)
while True:
    cv2.imshow("Imagem colorida",imagem)

    if cv2.waitKey(5) == 27:
        break

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
print(imagem_cinza.shape)

while True:
    cv2.imshow("Imagem convertida para cinza", imagem_cinza) #cv2.imshow("Titulo",imagem) Função para exibir a imagem
    if cv2.waitKey(5) == 27:
        break

detector_facial = cv2.CascadeClassifier('detection_recogntion/haarcascade_frontalface_default.xml')
deteccoes = detector_facial.detectMultiScale(imagem_cinza)
print(deteccoes) # os primeiros parametros são o eixo x e y, e os dois
                 #ultimos são as dimenções ou o tamanho da face detectada

qtdFaces = len(deteccoes)
print(qtdFaces)
for (x,y,w,h,) in deteccoes:
    print(x,y,w,h)
    cv2.rectangle(imagem, (x,y), (x + w, y + h), (0, 255, 255), 2)
while True:
    cv2.imshow("Face reconhecida",imagem)
    if cv2.waitKey(5) == 27:
        break

#Redimensionamento da imagem
imagem1 = cv2.imread("face/people1.jpg")
print(imagem1.shape)
imagem1_cinza = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
print(imagem1_cinza.shape)

while True:
    cv2.imshow("Pessoas cinza",imagem1_cinza)
    if cv2.waitKey(5) == 27:
        break

novaDeteccoes = detector_facial.detectMultiScale(imagem1_cinza)
print(novaDeteccoes)
print(len(novaDeteccoes))
for (x,y,w,h) in novaDeteccoes:
    cv2.rectangle(imagem1, (x,y), (x + w, y +h), (0, 255, 255), 2)
while True:
    cv2.imshow("Faces reconhecidas", imagem1)
    if cv2.waitKey(5) == 27:
        break
#para corrigir o problema de falsos positivos podemos reduzir a dimensionalidade da imagem
