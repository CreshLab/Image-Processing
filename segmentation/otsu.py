import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def otsu_thresholding(image):
    # Compute image histogram
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    
    # Compute grey-level probability
    pixel_total = image.shape[0] * image.shape[1]
    current_max, threshold = 0, 0
    sumT, sumF, sumB = 0, 0, 0
    weightB, weightF = 0, 0
    
    for i in range(0, 256):
        sumT += i * histogram[i]
    
    for i in range(0, 256):
        weightB += histogram[i]
        if weightB == 0:
            continue
        
        weightF = pixel_total - weightB
        if weightF == 0:
            break
        
        sumB += i * histogram[i]
        sumF = sumT - sumB
        
        meanB = sumB / weightB
        meanF = sumF / weightF
        
        # Compute inter-class variance
        var_between = weightB * weightF * (meanB - meanF) ** 2
        
        # Update treshold if the current variance value is greater than the previous one
        if var_between > current_max:
            current_max = var_between
            threshold = i
            
    return threshold

# Upload greyscale image
image = cv.imread('coins.png', cv.IMREAD_GRAYSCALE)

# Check if the image has been correctly read
assert image is not None, "file could not be read, check with os.path.exists()"

# Compute Otsu thresholding
threshold = otsu_thresholding(image)
print(f'Threshold calcolata da Otsu: {threshold}')

# Threshold applied to the image
_, binary_image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)

# Show original and binary image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Image binarized with Otsu')
plt.show()
