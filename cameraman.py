import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

image_path = Path('assets', 'cameraman.bmp')

imagem = cv2.imread(image_path.as_posix(), cv2.IMREAD_GRAYSCALE)

h1 = np.array([[1, 1, 1]])/3  
h2 = np.array([[1], [1], [1]])/3

intermediario = cv2.filter2D(imagem, -1, h1)
saida_1 = cv2.filter2D(intermediario, -1, h2)

filtro_combinado = np.ones((3,3))/9
saida_2 = cv2.filter2D(imagem, -1, filtro_combinado)


plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.title("Imagem Original")
plt.imshow(imagem, cmap="gray")

plt.subplot(1, 4, 2)
plt.title("h1")
plt.imshow(intermediario, cmap="gray")

plt.subplot(1, 4, 3)
plt.title("h2*(h1*Im)")
plt.imshow(saida_1, cmap="gray")

plt.subplot(1, 4, 4)
plt.title("(h1*h2)*Im")
plt.imshow(saida_2, cmap="gray")

plt.show()