import cv2
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

image_path = Path('assets', 'alumgrns.bmp')

image = cv2.imread(image_path.as_posix(), cv2.IMREAD_GRAYSCALE)

ksize = 31  
frequencies = [0.1, 0.2, 0.3, 0.4]  
orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  

# Gabor filter
responses = []
for theta in orientations:
    for frequency in frequencies:
        kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        responses.append(filtered)

# response to array
feature_vector = np.array(responses).reshape(len(responses), -1).T

n_clusters = 5 
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feature_vector)
labels = kmeans.labels_.reshape(image.shape)

unique_regions = np.unique(labels)
num_regions = len(unique_regions)

print(f"regions with different textures: {num_regions}")

label_image = (labels / num_regions * 255).astype('uint8')
cv2.imshow('Segmented Textures', label_image)
cv2.waitKey(0)
cv2.destroyAllWindows()