import os
import glob
import numpy as np 
import cv2 as cv
from matplotlib import pyplot as plt
import math
import numpy.linalg as alg
from mpl_toolkits.mplot3d import Axes3D


# Declaring parameters
k =np.array([[924.72662963,0,616.02979505],
 			[0,924.39978726,360.19539382],
 			[0,0,1]])
feature_params = dict( maxCorners = 4000,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
lk_params = dict(winSize=(21,21),maxLevel = 2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
pnp_params = dict(distCoeffs =np.zeros((5, 1)),iterationsCount = 100,reprojectionError = 8.0,confidence = 0.99,flags = 1)

# triangulate 
def triangulate(R1,t1,R2,t2,kp1,kp2,k):
    '''kp1  = kp1.astype(float)
    kp2  = kp2.astype(float)'''
    P1 = np.dot(k,np.hstack((R1,t1)))
    P2 = np.dot(k,np.hstack((R2,t2)))
    kp_1 = kp1.T
    kp_2 = kp2.T
    cloud = cv.triangulatePoints(P1, P2, kp_1,kp_2)
    print(cloud.shape)
    cloud= cloud /(cloud[3,:]+1e-8)
    return cloud.T[:,:,:3]

# find keypoint matches between two images given their descriptors and keypoints
def keypoint_matches(kp1,des1,kp2,des2):

    # create BFMatcher object
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

    # Match descriptors.
    bf = cv.BFMatcher()

    # convert keypoints to (x,y) coordinates
    p1 = cv.KeyPoint_convert(kp1)
    p2 = cv.KeyPoint_convert(kp2)

    # find matches
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # find good matches and corresponding points
    p11 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    des1 = np.array([ des1[m.queryIdx] for m in good ])
    p21=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    des2 = np.array([ des2[m.trainIdx] for m in good ])
    
    # return good matches, corresponding points and descriptors
    return good,p11,p21,des1,des2


# detect keypoints and descriptors
def feature_detection(img):

    # create FAST detector
    fast = cv.FastFeatureDetector_create()
    fast.setThreshold(10)
    print(fast.getThreshold())
    kp = fast.detect(img,None)

    # create ORB descriptor
    orb = cv.ORB_create()
    kp, des = orb.compute(img, kp)  
    return kp,des

# find reprojection error between 2d points and projection of their corresponding 3d points on the image plane
def reprojection_error(mat,P,pts):
    print(P.shape, mat.T.shape)
    pt3 = (P@mat.T).T
    pt3= pt3 / pt3[:,2].reshape((-1,1))
    pt3 = pt3[:,:2]
    rep = pt3 - pts.reshape((-1,2))
    rep = np.linalg.norm(rep)
    rep = rep/len(pts)
    return rep

# find the projection matrix of the camera for a given rotation and translation
def proj(k,R,t):
    Rt = np.zeros((3,4))
    Rt[:3,:3] = R
    Rt[:,3] = t.reshape((3))
    P = k@Rt
    return P

# find the rotation and translation between two images given their keypoints and descriptors
def relpose(kp1,kp2,k,R_old,t_old):

    # find fundamental matrix
    F, mask = cv.findFundamentalMat(kp1,kp2,cv.FM_RANSAC,0.6,0.9)
    kp1 = kp1[mask.ravel()==1]
    kp2 = kp2[mask.ravel()==1]

    # find essential matrix
    E = (k.T)@F@k

    # find rotation and translation
    points, R_est, t_est, mask_pose = cv.recoverPose(E, kp1,kp2,k)

    # update rotation and translation
    t_new = -(np.dot(R_old,t_est)).T
    R_new =R_old@R_est

    # return rotation, translation and inliers
    return t_new.T,R_new,kp1,kp2

# construct scene from the images
def constr_scene(cloud,color,name):
    color = color.reshape(-1, 3)
    fid = open(name+'.ply','wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%cloud.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    print(cloud.shape[0])
    for i in range(0,cloud.shape[0]):
      fid.write(bytearray(struct.pack("fffccc",cloud[i,0],cloud[i,1],cloud[i,2],color[i,2].tobytes(),color[i,1].tobytes(),color[i,0].tobytes())))
    fid.close()

# main function
# performs feature detection, matching, relative pose estimation and triangulation
def sfm(path,im):
    cloud = []
    color = []
    trajectory = []
    avail = False

   
    img_color = cv.imread(path +'/'+ im[0])
    img0 = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    kp0,des0 = feature_detection(img0)
    img1 = cv.imread(path +'/'+ im[1])

    # find keypoints and descriptors
    kp1,des1 = feature_detection(img1)

    # find matches
    good,common_pts0,common_pts1,common_des0,common_des1 = keypoint_matches(kp0,des0,kp1,des1) 
    
    # find essential matrix
    E,mask = cv.findEssentialMat(common_pts0,common_pts1,k,cv.RANSAC,0.999,2.0,1599)
    common_pts0 = common_pts0[mask.ravel() == 1]
    common_pts1 = common_pts1[mask.ravel() == 1]
    common_des0 = common_des0[mask.ravel() == 1]
    common_des1 = common_des1[mask.ravel() == 1]


    # find rotation and translation
    retval, R, t, mask = cv.recoverPose(E, common_pts0, common_pts1, k)
    common_pts0 = common_pts0[mask.ravel() == 255]
    common_pts1 = common_pts1[mask.ravel() == 255]
    common_des0 = common_des0[mask.ravel() == 255]
    common_des1 = common_des1[mask.ravel() == 255]

    # find projection matrix
    P1 = proj(k,np.eye(3).astype(np.float32),np.zeros((3,1))) 
    P2 = proj(k,R,t)

    # triangulate points to find 3d points
    point_cloud = triangulate(np.eye(3).astype(np.float32),np.zeros((3,1)),R,t, common_pts0, common_pts1,k)
    
    # print(0,reprojection_error(point_cloud,P1,common_pts0))
    # print(1,reprojection_error(point_cloud,P2,common_pts1))
    
    # add points to the scene
    mat = point_cloud
    print(point_cloud.shape)
    idx = common_pts0.reshape(-1,2).astype(np.uint8())
    color = img_color[idx[:,0],idx[:,1]]
    Rold = R
    told=t
    #print(color_data)
    for i in range(2,len(im)):
        img1_color = cv.imread(path +'/'+ im[i-1])
        img2 = cv.imread(path +'/'+ im[i],cv.IMREAD_GRAYSCALE)

        # find keypoints and descriptors
        kp2,des2 = feature_detection(img2)

        # find matches
        good,common_pts1,common_pts2,common_des1,common_des2 = keypoint_matches(common_pts1,common_des1,kp2,des2)
        #point_cloud = np.array([point_cloud[n.queryIdx,:,:] for n in good ])
        print(good)
        print(point_cloud.shape)

        # find relative pose between two images using RANSAC
        retval, rvec, t1, inliers = cv.solvePnPRansac(point_cloud,common_pts2, k, (0,0,0,0),useExtrinsicGuess = True ,iterationsCount = 70,reprojectionError = 4.5,flags = cv.SOLVEPNP_ITERATIVE)
        R1,_ = cv.Rodrigues(rvec)

        # update rotation and translation
        trajectory.append(t1)

        # find projection matrix
        P3 = proj(k,R1,t1)
        good,common_pts1,common_pts2,common_des1,common_des2 = keypoint_matches(kp1,des1,kp2,des2)

        # triangulate points to find 3d points
        point_cloud = triangulate(Rold,told, R1,t1, common_pts1, common_pts2,k)
        # print(reprojection_error(point_cloud,P3,common_pts2))


        idx = common_pts1.reshape(-1,2).astype(np.uint8())

        # add points to the scene
        mat = np.vstack((mat,point_cloud))

        # add color to the added points
        color = np.vstack((color,img1_color[idx[:,0],idx[:,1]]))
        img1 = img2
        kp1,des1 = kp2,des2
        kp1,common_pts1,common_des1 = kp2,common_pts2,common_des2
     
        P2 = P3
        Rold = R1
        told = t1
        i= i + 1

    # construct point cloud
    constr_scene(mat[:,:3],color.reshape(-1,3),"statue")
    plt.figure()
    plt.show()


path = "/home/mst3/Desktop/images_vo/hh"
L = os.listdir(path)
L.sort()
sfm(path,L)
