import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

image_path = Path('assets', 'cameraman.bmp')

image = cv2.imread(image_path.as_posix(), cv2.IMREAD_GRAYSCALE)

h1 = np.array([[1, 1, 1]])/3  
h2 = np.array([[1], [1], [1]])/3

intermediary = cv2.filter2D(image, -1, h1)
output_1 = cv2.filter2D(intermediary, -1, h2)

combined_filter = np.ones((3,3))/9
output_2 = cv2.filter2D(image, -1, combined_filter)


plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1, 4, 2)
plt.title("h1")
plt.imshow(intermediary, cmap="gray")

plt.subplot(1, 4, 3)
plt.title("h2*(h1*Im)")
plt.imshow(output_1, cmap="gray")

plt.subplot(1, 4, 4)
plt.title("(h1*h2)*Im")
plt.imshow(output_2, cmap="gray")

plt.show()