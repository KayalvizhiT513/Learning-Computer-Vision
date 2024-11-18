import cv2
import numpy as np
import os
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (5,8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)#for z only, directly assing mgrid value through error.

objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)#Making z=0(Chessboard kept stationary.)
#objp=objp*10
print(objp)

# Extracting path of individual image stored in a given directory
images = glob.glob('*.jpg')
for fname in images:
    img = cv2.imread(fname)
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)

    #resize the image to fit into the window.
    # Resize the image to a fixed width and height
    desired_width = 800  # Change to your desired width
    desired_height = 600  # Change to your desired height
    resized_img = cv2.resize(img, (desired_width, desired_height))
    
    cv2.imshow('img',resized_img)
    cv2.waitKey(50)

cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("Re-projection Error:", ret)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("number of rvecs : ",len(rvecs))
print("size of each rotation vector:",rvecs[0].shape)
print("tvecs : \n")
print(tvecs)
print("number of tvecs:",len(tvecs))
print("size of each rotation vector:",tvecs[0].shape)
print("Number of objects points : ",len(objpoints))
print("Number of objects points : ",len(imgpoints))

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "Average error: {}".format(mean_error/len(objpoints)) )
print("Focal Lengths:")
print(f"  Focal Length (fx): {mtx[0, 0]} pixels")
print(f"  Focal Length (fy): {mtx[1, 1]} pixels")