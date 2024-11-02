import cv2
import numpy as np
from pathlib import Path

folder_path = Path('assets', 'CME_11')
template_tool_path = Path('assets', 'CME_11', 'CME_11_1.jpg')
match_threshold = 5 
distance_threshold = 30 

template = cv2.imread(template_tool_path, 0)
template_keypoints, template_descriptors = cv2.ORB_create().detectAndCompute(template, None)

def contains_instrument(path):
    image = cv2.imread(path, 0)
    _, descriptors = cv2.ORB_create().detectAndCompute(image, None)

    if descriptors is None:
        return False  

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(template_descriptors, descriptors)
    
    if len(matches) > match_threshold:
        avg_distance = np.mean([match.distance for match in matches])
        if avg_distance < distance_threshold:
            return True
    return False

def is_blurry(path, blur_threshold=50):
    image = cv2.imread(path, 0)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < blur_threshold

def get_orientation(image_path):
    image = cv2.imread(image_path, 0)
    _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        _, _, angle = cv2.minAreaRect(cnt)
        
        if -10 <= angle <= 10:
            return "Horizontal"
        elif 80 <= abs(angle) <= 100:
            return "Vertical"
        else:
            return "Inclined"
    return None

for image_file in folder_path.glob("*.jpg"):
    if is_blurry(image_file):
        print(f"{image_file.name}: Blurred image")
    else:
        if contains_instrument(image_file):
            orientation = get_orientation(image_file)
            print(f"{image_file.name}: Contains instrument - Orientation: {orientation}")
        else:
            print(f"{image_file.name}: Does not contain instrument")