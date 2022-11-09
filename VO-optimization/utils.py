import numpy as np 
import scipy
import cv2 as cv
import pandas as pd
import torch


def getGT(path):
    """
    Return the ground truth

    @param path: path to the *.txt file

    Returns Ground Truth array
    """
    data = pd.read_csv(path,delimiter = " ")

    arr = data.to_numpy()

    return arr[:,:4]

def getScale(arr,id):
    """
    Returns the scale

    @param arr: array of groundtruth
    @param id: frame id

    Returns scale
    """
    return arr[-1,id]

def getEstimateScale(poses):
    prev_pose = poses[-2]
    cur_pose = poses[-1]

    return np.linalg.norm((cur_pose - prev_pose))

def get_TRANSF_mat(cur_pt):
    """
    Finds the transformation matrix so as to normalize the points with mean = 0 and scaled by sqrt(2)/std deviation

    @param: Array of points (3 * N)

    Returns transformation matrix
    """
    cur_pt = cur_pt / cur_pt[2]

    cur_mean = np.mean(cur_pt[:2], axis=1)
    cur_scale = 1 / max(cur_pt.shape)

    cur_TRANSF = np.array([[cur_scale, 0, 0],
                           [0, cur_scale, 0],
                           [0, 0, 1]])

    return cur_TRANSF


def normalize(cur_pt,cur_TRANSF):
    """
    Normalize the points based on the transformation matrix 

    @param cur_pt:  Keypoints from current frame (3 * N)
    @param cur_TRANSF: Transformation Matrix used to normalize points in current frame (3 * 3)

    Returns transformed points
    """
    return (cur_TRANSF.T @ cur_pt)

def unnormalize(F, prev_TRANSF, cur_TRANSF):
    """
    Unnormalize the Fundamental Matrix

    @param F: Normalized Fundamental Matrix (3 * 3)
    @param prev_TRANSF: Transformation Matrix used to normalize points in previous frame (3 * 3)
    @param cur_TRANSF: Transformation Matrix used to normalize points in current frame (3 * 3)

    Returns Unnormalized Fundamental Matrix
    """
    return cur_TRANSF.T @ F @ prev_TRANSF



def to_homogenous(pts):
    """
    Convert cartesian coordinates to homogenous coordinates

    @param pts: Array of points in cartesian coordinate system (2 * N)

    Returns points in homogenous coordinates (3 * N)
    """
    x,y = pts.shape
    return np.vstack((pts,np.ones((1,y)))) 

def to_homogenous_torch(pts):
    """
    Convert cartesian coordinates to homogenous coordinates

    @param pts: Array of points in cartesian coordinate system (2 * N)

    Returns points in homogenous coordinates (3 * N)
    """
    x,y = pts.shape
    return torch.cat((pts,torch.ones((1,y)).cuda()),dim=0) 

def from_homogenous(pts):
    """
    Convert points from homogenous coordinates to cartesian coordinates

    @param pts: Array of points in homogenous coordinates (3 * N)
    
    Returns points in cartesian coordinates (2 * N)
    """

    return pts[:2,:]/pts[2,:]