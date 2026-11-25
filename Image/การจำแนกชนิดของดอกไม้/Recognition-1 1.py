import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Extracts 'Color' features using a Color Histogram in the HSV color space.
    """
    # Convert image from BGR to HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate Histogram for Hue, Saturation, and Value channels
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    
    # Normalize the histogram so it is independent of image size
    cv2.normalize(hist, hist)
    
    # Flatten into a 1D vector (Size: 8x8x8 = 512 dimensions)
    return hist.flatten()

def extract_texture_glcm(gray_image):
    """
    Extracts 'Texture' features using GLCM (Gray-Level Co-occurrence Matrix).
    """
    # Compute GLCM matrix (distance = 1 pixel, angle = 0 degrees)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Extract structural texture properties from GLCM
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return np.array([contrast, homogeneity, energy, correlation])

def extract_shape_hu_moments(gray_image):
    """
    Extracts 'Shape' features using Hu Moments.
    """
    # Calculate spatial and central moments of the image
    moments = cv2.moments(gray_image)
    
    # Compute 7 Hu Moments
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Apply Log Transformation to scale down extreme variations
    for i in range(0, 7):
        if hu_moments[i] != 0:
            hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))
        else:
            hu_moments[i] = 0
            
    return hu_moments

def extract_combined_features(image_path):
    """
    Combines Color, Texture, and Shape extractions into a single merged feature vector.
    """
    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Create a grayscale copy for Texture and Shape processors
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Process individual pipelines
    color_feats = extract_color_histogram(img)
    texture_feats = extract_texture_glcm(gray)
    shape_feats = extract_shape_hu_moments(gray)
    
    # Concatenate features into a unified 1D profile vector
    features = np.hstack([color_feats, texture_feats, shape_feats])
        
    return features


def feature_extraction(image_dir):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if os.path.exists(image_dir):
       image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
        
    else:
       print(f"Error: Directory '{image_dir}' does not exist.")
       image_files = []
   
    feature_list = []
    labels_list = []
    numerical_features = []

    for filename in image_files:
        full_path = os.path.join(image_dir, filename)
        try:
            # Run composite fusion calculation
            features = extract_combined_features(full_path)
            if features is not None:
                 
                # Store structural mapping inside the main list container
                feature_list.append(features)
                
                                
                # Print the extracted feature values and isolated type label
                print(f"[SUCCESS] Processed: {filename}")
                
        except Exception as e:
            print(f"[ERROR] Skipping corrupt or unreadable image {filename}: {e}")

    print("Extraction completed")
    print("\n--------------------------------")
    return feature_list


# --- Call feature extraction function ---
training_features = []
training_labels = []
 
# 1. วนลูปดึงฟีเจอร์จากโฟลเดอร์ flower1 ถึง flower4
for i in range(1, 5):
    folder_path = rf"D:\ND107\Image\w11\flower-datasets\flower{i}"
    features = feature_extraction(folder_path)
    for f in features:
        training_features.append(f)
        training_labels.append(i)
 
# 2. ดึงฟีเจอร์รูปทดสอบ
test_feat = extract_combined_features(r"D:\ND107\Image\w11\flower-datasets\test_image.jpg")
 
# 3. คำนวณหาระยะห่างที่น้อยที่สุด (Nearest Neighbor)
distances = [np.linalg.norm(test_feat - tf) for tf in training_features]
best_match_index = np.argmin(distances)
predicted_class = training_labels[best_match_index]
 
print("\n=== Nearest Neighbor Result ===")
print(f"ดอกไม้ในรูปเป็นชนิด: flower{predicted_class}")
print(f"Distance: {distances[best_match_index]:.4f}")

 
