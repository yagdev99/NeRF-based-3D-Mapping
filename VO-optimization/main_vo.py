from pickle import NONE
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from IPython.display import display, clear_output
import os
from utils import from_homogenous, getGT, getScale, to_homogenous
from fundamental_matrix import RANSAC, recoverPose
from time import time
import torch
import cupy as cp


class VisualOdometry(object):
    def __init__(self, path, K, DETECTOR = cv.FastFeatureDetector_create(), DESCRIPTOR = cv.SIFT_create(), MATCHER = cv.BFMatcher(), gt_file = "groundtruth.txt",useOptFlow = False, precision = np.float32):
        """
        @param path: path to the dataset
        @param K: camera matrix
        @param DETECTOR: feature detector
        @param DESCRIPTOR: feature descriptor
        @param MATCHER: feature matcher
        @param gt_file: ground truth file (*.txt)
        @param useOptFlow: use optical flow or not
        """

        self.poses = []

        self.path = path
        self.K = K
        self.DETECTOR = DETECTOR
        self.DESCRIPTOR = DESCRIPTOR
        self.MATCHER = MATCHER
        # self.gt = getGT(os.path.join(self.path,gt_file)).T
        self.useOptFlow = useOptFlow

        # Initialize the rotation and translation 
        self.R = np.eye(3)
        self.t = np.zeros((3,1))

        self.precision = precision

        # Keeps track of the frame id
        self.frame_id = 0
        self.times = {}

        # Initialize the previous frame and the current frame
        self.cur_frame = None
        self.prev_frame = None

        # Initialize previous keypoint and descriptor
        self.prev_kp = None
        self.prev_des = None

        # Initialize current keypoint and descriptor
        self.cur_kp = None
        self.cur_des = None

        # Initialize array of corresponding points in current and previous frame
        self.prev_pt = np.zeros((3,1))
        self.cur_pt = np.zeros((3,1))

        # Creating subplots
        self.fig, (self.ax1,self.ax2) = plt.subplots(2,1)

        # Image to plot the trajectories
        self.traj = np.zeros(shape=(800, 600, 3))

        self.init_vo()

    
    def init_vo(self):
        """
        Initialize the Visual Odometry
        """

        # Read the first frame and intialize the kp, des and frame
        self.cap = cv.VideoCapture(os.path.join(self.path,"custom_data.mp4"))
        self.frame_id = 1
        ret, frame = self.cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if ret == True:
            self.prev_frame = self.cur_frame = frame
            self.pres_kp, self.prev_des = self.cur_kp, self.cur_des = self.detect_compute(frame)

        # Start the loop
        self.main()


    def detect_compute(self,frame,useOptFlow = False):
        """
        Detect and compute the features

        @param frame: current frame
        @param useOptFlow: use optical flow or not

        @return: keypoints and descriptors (if not using optical flow)
        """

        # detect the keypoints in the frame
        kp = self.DETECTOR.detect(frame, None)
        if not useOptFlow:

            # compute the descriptors for the keypoints in frame 
            kp, des = self.DESCRIPTOR.compute(frame, kp)
            return kp, des
        else: 
            return kp, None
    
    def matcher(self, useOptFlow = False, thres = 2000):
        """
        Featuring matching for the current and previous frame

        @param useOptFlow: use optical flow or not
        @param thres: threshold for the keypoints for optical flow

        Returns corresponding matches in current and previous frame
        """
        # Matching using optical flow
        if useOptFlow:

            # Using Lucas-Kanade optical flow

            # Parameters for lucas kanade optical flow
            lk_params = dict(winSize = (15, 15),
                  maxLevel = 5,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                              10, 0.05))
            
            # If number of points drop below threshold, then detect keypoints again
            if max(self.prev_pt.shape) < thres:
                self.prev_kp = self.detect_compute(self.prev_frame, useOptFlow)
                self.prev_pt = to_homogenous(np.asanyarray([np.array(pt.pt) for pt in self.prev_kp[0]]).T)
            
            
            # Find feature matches using LK Optical Flow
            self.cur_pt, st, err = cv.calcOpticalFlowPyrLK(self.prev_frame.astype(np.uint8), self.cur_frame.astype(np.uint8), from_homogenous(self.prev_pt).T.astype(np.float32), None, **lk_params)
            self.prev_pt, st, err = cv.calcOpticalFlowPyrLK(self.cur_frame.astype(np.uint8), self.prev_frame.astype(np.uint8), self.cur_pt.astype(np.float32), None, **lk_params)
           
            # st = 1 when corresponding match is found else st = 0cc
            # Getting the matches
            self.cur_pt = self.cur_pt[st[:,0] == 1]
            self.prev_pt = self.prev_pt[st[:,0] == 1]

                
        else:
            # Matching using Brute Force Matcher

            self.cur_pt = []
            self.prev_pt = []
        
            matches = self.MATCHER.knnMatch(self.prev_des,self.cur_des,2)
            
            # Applying Lowe's test
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    self.cur_pt.append(self.cur_kp[m.trainIdx].pt)
                    self.prev_pt.append(self.prev_kp [m.queryIdx].pt)
            
        return self.prev_pt, self.cur_pt

    def display(self, continous,plot_gt, plot):
        """
        Display the trajectory 

        @param continous: display the trajectory for each frame or not
        @param plot_gt: plot the ground truth or not
        @param plot: plot the trajectory using cv2 or not

        Returns the trajectory image 
        """
        pose = np.array(self.poses).reshape(-1,3).T
        x,y,z = pose[0,:], pose[1,:], pose[2,:]
        # x_gt, y_gt, z_gt = self.gt[0], self.gt[1], self.gt[2]

        if plot:
            if continous:
                self.ax1.plot(x,z,'g')
                # if plot_gt:
                    # self.ax1.plot(x_gt[:self.frame_id],z_gt[:self.frame_id])
                self.ax2.imshow(self.cur_frame)
                display(self.fig)    
                clear_output(wait = True)
                plt.pause(0.1) 
            else:
                self.ax1.plot(x,z,'g')
                self.ax2.plot(x,y,'b')
                # if plot_gt: 
                    # self.ax1.plot(x_gt,z_gt)
                    # self.ax2.plot(x_gt,y_gt)
                
                plt.show()

            return None

        else:
            draw_x, draw_y, draw_z = [-1*int(round(x)) for x in pose[:,-1]]
            # true_x, true_y, true_z = [int(round(x)) for x in self.gt[:3,self.frame_id-2]]

            # traj = cv.circle(self.traj, (true_x + 300, true_z + 400), 1, list((0, 0, 255)), 3)
            traj = cv.circle(self.traj, (draw_x + 300, draw_z + 400), 1, list((0, 255, 0)), 3)

            cv.putText(traj, "Ground Truth: ", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, )
            cv.putText(traj, "-------", (150, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv.putText(traj, "Visual Odometry: ", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, ) 
            cv.putText(traj, "-------", (150, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, )

            cv.imshow("self.traj", traj)
            cv.imshow("image",self.cur_frame)
            cv.imwrite("traj.jpg",traj)
            cv.waitKey(1)

            return traj

    # main loop for visual odometry
    def main(self):
        clahe = cv.createCLAHE()
        self.times["feature_detector"] = 0
        self.times["F_calc"] = 0
        self.times["recoverPose"] = 0
        self.times["plot_time"] = 0

        while(self.cap.isOpened()):

            # reading each frame
            ret, frame = self.cap.read()

            # TODO: Apply CLAHE for better feature detection 
            print(ret)

            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    

            if ret == True:

                self.cur_frame = frame

                prev = time()

                # Feature Matching
                if self.useOptFlow:
                    self.cur_kp, self.cur_des = self.detect_compute(self.cur_frame)
                self.prev_pt, self.cur_pt = self.matcher(self.useOptFlow)


                # pushing to gpu 
                self.prev_pt, self.cur_pt = np.array(self.prev_pt,dtype=self.precision).T , np.array(self.cur_pt,dtype=self.precision).T
                self.prev_pt, self.cur_pt = to_homogenous(self.prev_pt), to_homogenous(self.cur_pt)

                self.times["feature_detector"] += time() - prev

                # Compute the fundamental matrix and inliers using RANSAC
                prev = time()
                # F, mask = cv.findFundamentalMat(from_homogenous(self.cur_pt).T,from_homogenous(self.prev_pt).T, cv.FM_RANSAC, 0.4, 0.999,200,None)
                F, mask = RANSAC(self.prev_pt, self.cur_pt, iterations = 200, threshold = 0.01, N = 8, precision = self.precision)
                E = self.K.T @ F @ self.K

                self.times["F_calc"] += time() - prev

                # Essential Matrix from Fundamental Matrix
                # Decompose the essential matrix and find R and t
                prev = time()
                if self.frame_id < 2:              
                    
                    # _,self.R, self.t, _ = cv.recoverPose(E,from_homogenous(self.prev_pt[:,mask.ravel() == 1]).T, from_homogenous(self.cur_pt[:,mask.ravel() == 1]).T,cameraMatrix= self.K)
                    self.R, self.t = recoverPose(E,self.cur_pt[:,mask], self.prev_pt[:,mask],self.K)
                    self.poses.append(self.t.copy())

                else: 
                   
                    # _, R, t, _ = cv.recoverPose(E,from_homogenous(self.prev_pt[:,mask.ravel() == 1]).T, from_homogenous(self.cur_pt[:,mask.ravel() == 1]).T,cameraMatrix= self.K)
                    R, t = recoverPose(E,self.cur_pt[:,mask], self.prev_pt[:,mask],self.K)
                    
                    # Get scale from ground truth
                    # scale = getScale(self.gt,self.frame_id-3)
                    #  scale = getEstimateScale(self.poses)

                    # find the current rotation and translation
                    scale = 1
                    self.t = self.t + scale * self.R.dot(t)
                    self.R = R.dot(self.R)

                    # append to the list of poses to plot
                    self.poses.append(self.t.copy())
                
                self.times["recoverPose"] += time() - prev

                prev = time()            
                print("processed frame id:", self.frame_id)
                print(self.times)
                self.frame_id += 1

                self.prev_pt = self.cur_pt.copy()
                self.prev_frame, self.prev_kp, self.prev_des = self.cur_frame, self.cur_kp, self.cur_des

                # plot the trajectory
                img = self.display(continous= False, plot_gt = False, plot = False)
                self.times["plot_time"] += time() - prev

            cv.imshow("image",img)
            cv.waitKey(0)

if __name__ == "__main__":
    path = "/home/yagnesh/Desktop/FYP/images_vo"
    precision = np.float64
    K = np.array([  [7.215377000000e02, 0.000000000000e00, 6.095593000000e02],
                        [0.000000000000e00, 7.215377000000e02, 1.728540000000e02],
                        [0.000000000000e00, 0.000000000000e00, 1.000000000000e00]], dtype=precision)
    vo = VisualOdometry(path, K,useOptFlow=True, precision=precision)