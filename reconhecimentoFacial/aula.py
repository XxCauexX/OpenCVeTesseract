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
imagem = cv2.imread("face/people1.jpg")
print(imagem.shape)
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
print(imagem_cinza.shape)

while True:
    cv2.imshow("Pessoas cinza",imagem_cinza)
    if cv2.waitKey(5) == 27:
        break

novaDeteccoes = detector_facial.detectMultiScale(imagem_cinza)
print(novaDeteccoes)
print(len(novaDeteccoes))
for (x,y,w,h) in novaDeteccoes:
    cv2.rectangle(imagem, (x,y), (x + w, y +h), (0, 255, 255), 2)
while True:
    cv2.imshow("Faces reconhecidas", imagem)
    if cv2.waitKey(5) == 27:
        break

#para corrigir o problema de falsos positivos podemos reduzir a dimensionalidade da imagem
#podemos fazer para evitar a distorção é calcular a proporção que a imagem será reduzida
imagem = cv2.imread("face/people1.jpg")
nova_largura = 600
proporcao = 1680 / 1120
nova_altura = int(nova_largura / proporcao)

imagem_redimensionada = cv2.resize(imagem, (nova_largura, nova_altura))
print(imagem_redimensionada.shape)

while True:
    cv2.imshow("Imagem redimensionada", imagem_redimensionada)
    if cv2.waitKey(5) == 27:
        break

imagem_cinza = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2GRAY)

while True:
    cv2.imshow("Imagem redimensionada e cinza", imagem_cinza)
    if cv2.waitKey(5) == 27:
        break

novaDeteccao = detector_facial.detectMultiScale(imagem_cinza)
print(len(novaDeteccao))

for (x,y,w,h) in novaDeteccao:
    cv2.rectangle(imagem_redimensionada, (x,y), (x + w, y + h), (0, 255, 255), 2)

while True:
    cv2.imshow("Faces reconhecidas", imagem_redimensionada)
    if cv2.waitKey(5) == 27:
        break

#redimensionando por escala
imagem = cv2.imread("face/people1.jpg")
imagem = cv2.resize(imagem, (0, 0), fx=0.5, fy=0.5)
print(imagem.shape)

while True:
    cv2.imshow("Faces reconhecidas", imagem)
    if cv2.waitKey(5) == 27:
        break

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
novaDeteccao = detector_facial.detectMultiScale(imagem_cinza)

for (x, y, w, h) in novaDeteccao:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 255), 2)

while True:
    cv2.imshow("faces reconhecidas", imagem)
    if cv2.waitKey(5) == 27:
        break

#Parãmetros haarcascade
#scaleFactor - Quanto menor o valor, o algoritimo sera capaz de detectar faces menores
imagem = cv2.imread("face/people2.jpg")
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
deteccoes = detector_facial.detectMultiScale(imagem_cinza, scaleFactor=1.2)
for (x, y, w, h) in deteccoes:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 255), 2)

while True:
    cv2.imshow("", imagem)
    if cv2.waitKey(5) == 27:
        break
