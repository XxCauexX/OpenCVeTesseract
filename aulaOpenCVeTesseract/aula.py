import pytesseract
import cv2

# passo 1: Ler a imagem com o OpenCV
imagem = cv2.imread("print_magalu.JPG")

caminho = r"E:\Program Files\Tesseract-OCR"
# passo 2: pedir pro tesseract extrair o texto da imagem

pytesseract.pytesseract.tesseract_cmd = caminho + r"\tesseract.exe"
texto = pytesseract.image_to_string(imagem, lang="por")

print(texto)