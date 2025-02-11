import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
filepath = "D:\\Computer Vision\\Feature engineering\\Checkerboard.jpg"
# filepath = "D:\\Computer Vision\\Feature engineering\\left05.jpg"
# filepath = "D:\\Computer Vision\\Feature engineering\\lisa.jpg"
image = cv2.imread(filepath)

# Convert the image from BGR to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(gray_image)

num_of_rows = len(gray_image)
num_of_cols = len(gray_image[0])
kx = np.array([[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]])
ky = np.array([[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]])
kx = np.array([[-3, 0, 3],
              [-10, 0, 10],
              [-3, 0, 3]])
ky = np.array([[-3, -10, -3],
              [0, 0, 0],
              [3, 10, 3]])

edge_detected_img = []
pre_processed_img = []

for ridx in range(num_of_rows):
    edge_detected_img.append([])
    pre_processed_img.append([])    
    for cidx in range(num_of_cols):
        edge_detected_img[ridx].append(0)
        pre_processed_img[ridx].append(0)

for ridx in range(1, num_of_rows-1):
    for cidx in range(1, num_of_cols-1):
        pre_processed_img[ridx][cidx] = (0.8*gray_image[ridx][cidx] + (0.025*gray_image[ridx-1][cidx-1] + 0.025*gray_image[ridx-1][cidx] + 0.025*gray_image[ridx-1][cidx+1] +
                                                                       0.025*gray_image[ridx+1][cidx-1] + 0.025*gray_image[ridx+1][cidx] + 0.025*gray_image[ridx+1][cidx+1] +
                                                                       0.025*gray_image[ridx][cidx-1] + 0.025*gray_image[ridx][cidx+1]))/9

pre_processed = np.array(pre_processed_img)        
cv2.imwrite(r"D:\\Computer Vision\\Feature engineering\\preprocessedimage.jpg",pre_processed)

for ridx in range(1, num_of_rows-1):
    for cidx in range(1, num_of_cols-1):
        mat = np.array([[pre_processed_img[ridx-1][cidx-1], pre_processed_img[ridx-1][cidx], pre_processed_img[ridx-1][cidx+1]],
                        [pre_processed_img[ridx+1][cidx-1],pre_processed_img[ridx+1][cidx],pre_processed_img[ridx+1][cidx+1]],
                        [pre_processed_img[ridx][cidx-1],pre_processed_img[ridx][cidx+1],pre_processed_img[ridx][cidx]]])

        Gx = np.sum(np.multiply(kx, mat))
        Gy = np.sum(np.multiply(ky, mat))
  
        M = np.array([[Gx*Gx, Gx*Gy], 
                       [Gx*Gy, Gy*Gy]])
        
        R = np.linalg.det(M) - 0.06*(np.trace(M)**2)
        edge_detected_img[ridx][cidx] = -R

print("Maximum value of R: ", np.max(edge_detected_img)) 

threshold = 0.15*np.max(edge_detected_img)
for ridx in range(1, num_of_rows-1):
    for cidx in range(1, num_of_cols-1):
        if edge_detected_img[ridx][cidx] > threshold:
            # add circle on the image
            cv2.circle(image, (cidx, ridx), 10, (0, 0, 255), 5)

# Display the grayscale image
# image = cv2.convertScaleAbs(np.array(edge_detected_img))
plt.figure(dpi=100)  # Adjust DPI for size control
plt.imshow(image)
plt.axis('off')  # Hide axes
plt.show()
"""
cv2.imshow('Grayscale Image', cv2.convertScaleAbs(np.array(edge_detected_img)), aspect=0.5)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
        
        
