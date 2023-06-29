import cv2 # biblioteca do openCV
import numpy as np

print(cv2.__version__)

imagem = cv2.imread("person.png")
print(imagem.shape)
m = [1,2,3,4,5,6,7,8,9,0,9,8,7,6,5,4,3,2,1,3,4,5,6,7,7,8,9,7,6,5,4,3,4,56,2,7,8,3,]
for c in m :
    c+= 1
    cv2.imshow("teste",imagem)
    print("imagem")

