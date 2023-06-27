import pytesseract
import cv2

# passo 1: Ler a imagem com o OpenCV
imagem = cv2.imread("print-teste.png")
imagem1 = cv2.imread("print_magalu.JPG")

caminho = r"E:\Program Files\Tesseract-OCR"
# passo 2: pedir pro tesseract extrair o texto da imagem

pytesseract.pytesseract.tesseract_cmd = caminho + r"\tesseract.exe"
texto = pytesseract.image_to_string(imagem, lang="por")
texto1 = pytesseract.image_to_string(imagem1, lang="por")
print("--------------------------------")
print("-----------Teste 1--------------")
print(texto)
print("--------------------------------")
print("-----------Teste 2--------------")
print(texto1)