import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
filepath = "D:\\Computer Vision\\Feature engineering\\20241117_165731.jpg"
# filepath = "D:\\Computer Vision\\Feature engineering\\left05.jpg"
image = cv2.imread(filepath)

# Convert the image from BGR to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(gray_image)

num_of_rows = len(gray_image)
num_of_cols = len(gray_image[0])
k = np.array([[1, 1, 1],
              [1, -4, 1],
              [1, 1, 1]])

pixelwise_gradient = []
edge_detected_img = []
pre_processed_img = []

for ridx in range(num_of_rows):
    pixelwise_gradient.append([])
    edge_detected_img.append([])
    pre_processed_img.append([])    
    for cidx in range(num_of_cols):
        pixelwise_gradient[ridx].append(0)
        edge_detected_img[ridx].append(0)
        pre_processed_img[ridx].append(0)


for ridx in range(1, num_of_rows-1):
    for cidx in range(1, num_of_cols-1):
        pre_processed_img[ridx][cidx] = (0.8*gray_image[ridx][cidx] + (0.025*gray_image[ridx-1][cidx-1] + 0.025*gray_image[ridx-1][cidx] + 0.025*gray_image[ridx-1][cidx+1] +
                                                                       0.025*gray_image[ridx+1][cidx-1] + 0.025*gray_image[ridx+1][cidx] + 0.025*gray_image[ridx+1][cidx+1] +
                                                                       0.025*gray_image[ridx][cidx-1] + 0.025*gray_image[ridx][cidx+1]))/9
      
for ridx in range(1, num_of_rows-1):
    for cidx in range(1, num_of_cols-1):        
        Gx = np.matmul(k, np.array([pre_processed_img[ridx][cidx-1], pre_processed_img[ridx][cidx], pre_processed_img[ridx][cidx+1]]))
        Gy = np.matmul(k, np.array([pre_processed_img[ridx-1][cidx], pre_processed_img[ridx][cidx], pre_processed_img[ridx+1][cidx]]))
        G = np.sqrt(np.matmul(Gx, Gx.T) + np.matmul(Gy, Gy.T))
  
        pixelwise_gradient[ridx][cidx] = G

for ridx in range(1, num_of_rows-1):
    for cidx in range(1, num_of_cols-1):        
        #curr_pixel = 1.5 * pixelwise_gradient[ridx][cidx]
        #if curr_pixel < pixelwise_gradient[ridx-1][cidx-1] or curr_pixel < pixelwise_gradient[ridx-1][cidx] or curr_pixel < pixelwise_gradient[ridx-1][cidx+1] or curr_pixel < pixelwise_gradient[ridx][cidx-1] or curr_pixel < pixelwise_gradient[ridx][cidx+1] or curr_pixel < pixelwise_gradient[ridx+1][cidx-1] or curr_pixel < pixelwise_gradient[ridx+1][cidx] or curr_pixel < pixelwise_gradient[ridx+1][cidx+1]:
        #    edge_detected_img[ridx][cidx] = 255
        
        if 8.5*pixelwise_gradient[ridx][cidx] < (pixelwise_gradient[ridx-1][cidx-1] + pixelwise_gradient[ridx-1][cidx] + pixelwise_gradient[ridx-1][cidx+1] + pixelwise_gradient[ridx][cidx-1] + pixelwise_gradient[ridx][cidx+1] + pixelwise_gradient[ridx+1][cidx-1] + pixelwise_gradient[ridx+1][cidx] + pixelwise_gradient[ridx+1][cidx+1]):
            edge_detected_img[ridx][cidx] = 255
            
# Display the grayscale image
image = cv2.convertScaleAbs(np.array(edge_detected_img))
plt.figure(dpi=100)  # Adjust DPI for size control
plt.imshow(image)
plt.axis('off')  # Hide axes
plt.show()
"""
cv2.imshow('Grayscale Image', cv2.convertScaleAbs(np.array(edge_detected_img)), aspect=0.5)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
        
        
