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
#podemos fazer para evitar a distorção é calcular a proporção que a imagem será reduzida
imagem2 = cv2.imread("face/people1.jpg")
nova_largura = 600
proporcao = 1680 / 1120
nova_altura = int(nova_largura / proporcao)

imagem_redimensionada = cv2.resize(imagem2, (nova_largura, nova_altura))
print(imagem_redimensionada.shape)

while True:
    cv2.imshow("Imagem redimensionada", imagem_redimensionada)
    if cv2.waitKey(5) == 27:
        break

imgR_cinza = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2GRAY)

while True:
    cv2.imshow("Imagem redimensionada e cinza", imgR_cinza)
    if cv2.waitKey(5) == 27:
        break

novaDeteccao = detector_facial.detectMultiScale(imgR_cinza)
print(len(novaDeteccao))

for (x,y,w,h) in novaDeteccao:
    cv2.rectangle(imagem_redimensionada, (x,y), (x + w, y + h), (0, 255, 255), 2)

while True:
    cv2.imshow("Faces reconhecidas", imagem_redimensionada)
    if cv2.waitKey(5) == 27:
        break

#redimensionando por escala
imagem2 = cv2.imread("face/people1.jpg")
imagem2 = cv2.resize(imagem2, (0, 0), fx=0.5, fy=0.5)
print(imagem2.shape)

while True:
    cv2.imshow("Faces reconhecidas", imagem2)
    if cv2.waitKey(5) == 27:
        break

img_cinza = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)
novaDeteccao = detector_facial.detectMultiScale(img_cinza)

for (x, y, w, h) in novaDeteccao:
    cv2.rectangle(imagem2, (x, y), (x+w, y+h), (0, 255, 255), 2)

while True:
    cv2.imshow("faces reconhecidas", imagem2)
    if cv2.waitKey(5) == 27:
        break
