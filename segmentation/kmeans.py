import cv2
import numpy as np

image = cv2.imread('image.jpg') # insert your image
Z = image.reshape((-1, 3))
Z = np.float32(Z)
K = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape((image.shape))

cv2.imwrite('segmented_image.jpg', segmented_image)
