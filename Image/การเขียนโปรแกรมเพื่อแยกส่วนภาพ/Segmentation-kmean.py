import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load and resize the image
image = cv2.imread('coin.jpg')  # Replace with your image path
image = cv2.resize(image, (300, 300))  # Resize for faster processing

# Reshape the image to a 2D array of pixels (num_pixels, 3)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define number of clusters (segments)
k = 2
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)

# Apply K-Means clustering
labels = kmeans.fit_predict(pixel_values)

# Generate random colors for each cluster (for visual distinction)
custom_colors = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.uint8)

# Replace each pixel value with the custom color of its cluster
segmented_data = custom_colors[labels]
segmented_image = segmented_data.reshape(image.shape)

# Display the original and segmented images
cv2.imshow('Original Image', image)
cv2.imshow(f'Segmented Image with Distinct Colors (K={k})', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
