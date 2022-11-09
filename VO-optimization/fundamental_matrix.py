import numpy as np
import cupy as cp
import torch
import cv2 as cv

np.random.seed(0)
cp.random.seed(0)
torch.manual_seed(0)

def RANSAC(prev_pt, cur_pt, iterations = 100, threshold = 0.1, N = 8, precision = np.float32):
    """
    Finds the best estimate of Fundamental Matrix, inliear and the outliers using RANSAC. Error is calculated based on the distance from epipolar line.
    CuPy Implementation

    @param prev_pt: Array of points from previous frame (3 * N)
    @param cur_pt: Array of points from current frame (3 * N)

    Returns fundamental matrix, inliers, outliers
    """
    prev_pt_mean = np.mean(prev_pt,axis = 1)
    prev_pt_max = np.max(prev_pt)
    prev_pt_T = np.eye(3)/prev_pt_max
    prev_pt_T[0][2] = -prev_pt_mean[0]/prev_pt_max
    prev_pt_T[1][2] = -prev_pt_mean[1]/prev_pt_max
    prev_pt_T[2][2] = 1
    prev_pt_norm = prev_pt_T @ prev_pt

    cur_pt_mean = np.mean(cur_pt,axis = 1)
    cur_pt_max = np.max(cur_pt)
    cur_pt_T = np.eye(3)/cur_pt_max
    cur_pt_T[0][2] = -cur_pt_mean[0]/cur_pt_max
    cur_pt_T[1][2] = -cur_pt_mean[1]/cur_pt_max
    cur_pt_T[2][2] = 1
    cur_pt_norm = cur_pt_T @ cur_pt

    best_mask = None
    num_pts = prev_pt.shape[1]
    num = np.zeros((iterations,num_pts),dtype=precision)

    # Randomly select N=8 points from the previous frame and current frame
    # and assumes them to be inliers and finds the fundamental matrix using the 8-point algorithm
    # Now find the inliers and outliers using the Sampson distance as error measure
    # Store the best fundamental matrix (having maximum inliers) and mask of inliers
    # Repeat the process for certain iterations and return the best fundamental matrix

    # Matrix of random number. Each columns represents one iteration of RANSAC
    idx = np.random.randint(low = 0, high = num_pts-1, size = (N,iterations))
    
    # selecting random points for all iteration of RANSAC all at once 
    sel_prev_pt = prev_pt_norm[:,idx]
    sel_cur_pt = cur_pt_norm[:,idx]

    # Finding all fundamental matrices associated with 
    F_stacked = prev_pt_T.T @ estimateFundamentalMatrix(sel_prev_pt, sel_cur_pt, precision) @ cur_pt_T
    # F_stacked = estimateFundamentalMatrix(sel_prev_pt, sel_cur_pt, precision)

    # Finding Sampson's distance for the whole batch
    num = np.power(np.sum(prev_pt.reshape(1,3,-1) * (F_stacked @ cur_pt),axis = 1,dtype=precision),2)

    prev_epline = (F_stacked @ prev_pt)
    cur_epline = (F_stacked @ cur_pt)      

    dnom = (prev_epline[:,0]**2 + prev_epline[:,1]**2) + (cur_epline[:,0]**2 + cur_epline[:,1]**2)
    error = num/dnom

    # Mask for inliers for the batch
    mask = error < threshold

    # number of inliers per iteration stored in 1d tensor
    inliers = np.sum(mask,axis = 1)
    
    # Finding best F across all iterations
    best_F = F_stacked[np.argmax(inliers),:,:].reshape((3,3))
    best_mask = mask[np.argmax(inliers),:]

    return (best_F/best_F[2,2]), (best_mask)
 

def estimateFundamentalMatrix(prev_pt, cur_pt, precision = np.float32):
    """
    Calculated the Fundamental Matrix using the 8-point algorithm for the whole batch at once

    CuPy version 

    @param prev_pt: Keypoints from previous frame (3 * N)
    @param cur_pt:  Keypoints from current frame (3 * N)

    Returns the estimated Fundamental Matrix
    """
    # Solve A*f = 0 using least squares.
    # A is a matrix of Kronker Products of the corresponding points size (2*N, 9)
    # f is flattened array of the fundamental matrix size (9,1)

    r,c,d = prev_pt.shape
    # Finding the Kronker Product of the points
    A = np.kron(prev_pt.T,cur_pt.T)

    # Solving using least squares 
    step1 = np.arange(0,A.shape[0],d+1)
    step2 = np.arange(0,A.shape[1],c+1)
    
    U, S, V = np.linalg.svd(A[step1][:,step2,:])

    # Fundamental Matrix associated with smallest singular value
    F = V[:,-1,:].reshape(-1,3, 3)

    # Ensure that the fundamental matrix is rank 2 by forcing the last singular value to be zero.
    U, S, V = np.linalg.svd(F)
    S[:,2] = 0
    
    S = np.einsum('ij,ik->jik', S.T, np.eye(S.T.shape[0], dtype=S.dtype))
    # print(U.shape, S.shape, V.shape, S[12,1])
    F = np.matmul(U, np.matmul(S, V)) 

    return (F.T/ F.T[-1,-1,:]).T


def depth_score(prev_P,cur_P, X, thres  = -1.0):
    """
    Calculates the depth score of the points. Find the number of points in front 
    of the camera for the given projection matrixes and the triangulated points.

    @param prev_P: Projection matrix of the prev frame
    @param cur_P: Projection matrix of the cur frame
    @param X: Triangulated points
    @param thres: Threshold for depth

    Returns the depth score
    """

    # Projecting the triangulated points 
    x1 = prev_P @ X
    x2 = cur_P @ X

    # depth of points = z coordinate of the point in camera frame
    depth1 = x1[2,:] 
    depth2 = x2[2,:]

    # Find the number of points in front of the camera
    scores = (depth1 < 0.0) * (depth2 < 0.0) * (depth1 > thres) * (depth2 > thres)
    scores= np.sum(scores)

    return scores


def triangulate_pt(prev_P, cur_P, prev_inliers, cur_inliers):
    """
    Triangulates the points using the projection matrices and the inliers in 
    previous and current frame.

    @param prev_P: Projection matrix of the prev frame
    @param cur_P: Projection matrix of the cur frame
    @param prev_inliers: Inliers in prev frame (3 * N)
    @param cur_inliers: Inliers in cur frame (3* N)

    Returns the triangulated points
    """
    return cv.triangulatePoints(prev_P, cur_P, prev_inliers, cur_inliers)

def recoverPose(E, prev_inliers, cur_inliers, K):
    """
    Finds decomposes the essential matrix and finds the best rotation and translation.

    @param E: Essential Matrix (3 * 3)
    @param prev_inliers: Inliers in prev frame (3 * N)
    @param cur_inliers: Inliers in cur frame (3* N)
    @param K: Camera Matrix (3 * 3)

    """

    # Decomposing the essential matrix
    W = np.array([  [0,1,0],
                    [-1,0,0],
                    [0,0,1]], dtype= np.float64)

    U, D, V_T = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(V_T) < 0:
        V_T = -V_T
    
    # Recovered rotation matrixes
    R_1 = np.dot(U, np.dot(W, V_T))
    R_2 = np.dot(U, np.dot(W.T, V_T))

    t = U[:,2].reshape(-1,1)

    # To find the correct rotation matrix and translation matrix from 4 possibilities
    # Check the number of points lying in front of the camera for each possibility
    # and choose the one with maximum number of points

    prev_P = K @ np.concatenate((np.eye(3), np.zeros((3,1))),axis = 1)

    R_mat = [R_1, R_2]
    t_mat = [t, -t]

    # Projection matrixes assciated with each possibility of (R, t)
    cur_P_mat = []
    for i in range(2):
        for j in range(2):
            cur_P = K @ np.concatenate((R_mat[i], t_mat[j]),axis = 1)
            cur_P_mat.append(cur_P)

    #Triangulating the points associated with each possibility of (R, t)
    X_mat = []

    # To cartesian coordinates
    prev_inliers_cart = prev_inliers[:2]/prev_inliers[2]
    cur_inliers_cart = cur_inliers[:2]/cur_inliers[2]

    for i in range(4):
        X_mat.append(triangulate_pt(prev_P, cur_P_mat[i], prev_inliers_cart, cur_inliers_cart))

    # Finding the number of points in front of the camera for each possibility of (R, t)
    scores = []
    for i in range(4):
        scores.append(depth_score(prev_P, cur_P_mat[i], X_mat[i]))
    
    # Choosing the possibility with maximum number of points in front of the camera
    idx = np.argmax(np.array(scores))
    if idx == 0:
        return R_1,-t
    elif idx == 1:
        return R_1, t
    elif idx == 2:
        return R_2, -t
    else:
        return R_2, t
