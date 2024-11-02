import cv2
import numpy as np
from pathlib import Path

image_path = Path('assets', 'tools.bmp')
image = cv2.imread(image_path.as_posix(), cv2.IMREAD_GRAYSCALE)

_, thresh = cv2.threshold(image, 225, 255, cv2.THRESH_BINARY_INV)
kernal = np.ones((2, 2), np.uint8)

dilation = cv2.dilate(thresh, kernal, iterations=2)

contours, _ = cv2.findContours(
    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

tools = str(len(contours))
print(f'{tools} tools detected in image' )

cv2.imshow('Dilation', dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()


