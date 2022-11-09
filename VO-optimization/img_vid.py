import cv2
import numpy as np
import glob

img_array = []
img_arr = glob.glob('/home/yagnesh/Desktop/FYP/images_vo/*')
print(img_arr)
# exit(0)
for filename in sorted(img_arr):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()