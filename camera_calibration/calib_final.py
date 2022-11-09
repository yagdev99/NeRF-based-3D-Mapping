import numpy as np
import cv2 as cv
import glob
x = 8
y = 10

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x*y,3), np.float32)
objp[:,:2] = np.mgrid[0:x,0:y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

i = 0

# Stores all the images in the folder
images = glob.glob('/home/yagnesh/Desktop/FYP/camera_calibration/Camera_Roll/*.jpg')

###################################################################
# Block to find the calibration matrix and distortion coefficients
###################################################################

# Iterates through all the images in the folder and finds the corners of the chessboard
for fname in images:

    # Reads the image and converts it to grayscale
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners 
    ret, corners = cv.findChessboardCorners(gray, (x,y), None)

    i += 1

    # If found, add object points, image points (after refining them)
    if ret == True:
        
        objpoints.append(objp)

        # Refines the corners
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

        # Stores the refined corners
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (x,y), corners2, ret)

        # Displays the image with the corners
        # cv.imshow('img', img)
        # cv.imwrite("results/image_"+str(i)+".jpg", img)
        # cv.waitKey(500)


# Finds the camera matrix and distortion coefficients using the object and image points
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Prints the camera matrix and distortion coefficients
print(mtx)

#################################################
# Block to find the mean error of the calibration
#################################################
mean_error = 0
for i in range(len(objpoints)):
    # Finds the projection of the 3D points on the image plane using the camera matrix and distortion coefficients
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

    # Finds the error between the projected points and the actual points in the image plane 
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)

    # Accumulates the mean error for each object point
    mean_error += error


print( "total error: {}".format(mean_error/len(objpoints)) )

cv.destroyAllWindows()
