import numpy as np
import cv2 as cv
import glob
import time
import pynput

from pynput import keyboard

class MyException(Exception): pass

def on_press(key):
    if key == keyboard.Key.esc:
        raise MyException(key)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((11*9,3), np.float32)
objp[:,:2] = np.mgrid[0:11,0:9].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# images = glob.glob('*.jpg')
cap = cv.VideoCapture(6)
index = 1

now = round(time.time() * 1000)
prev = round(time.time() * 1000)
while 1:
    now = round(time.time() * 1000)



# Collect events until released
    with keyboard.Listener(
            on_press=on_press) as listener:
        try:
            listener.join()
        except MyException as e:
            print('{0} was pressed'.format(e.args[0]))
            cv.imwrite("img" + str(index) + ".png",img)
    
    # print(now)
    _, img = cap.read()
    # if now - prev > 5000:
    #     print(now)
    #     cv.imwrite("img" + str(index) + ".png",img)
    #     prev = round(time.time() * 1000)
    #     index += 1
        


    
    # print(type(img))

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    # ret, corners = cv.findChessboardCorners(gray, (11,9), None)
    # # If found, add object points, image points (after refining them)
    # if ret == True:
    #     print("here")
    #     objpoints.append(objp)
    #     corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    #     imgpoints.append(corners)
    #     # Draw and display the corners
    #     cv.drawChessboardCorners(img, (11,9), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(10)
    

cv.destroyAllWindows()
