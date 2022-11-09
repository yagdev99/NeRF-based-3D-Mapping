import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Triangulate the 2D points in x1, x2 under the two projection matrixes P1, P2.
def triangulation(corr, points2D, P1, P2):

    # remove points with negative x and y values
    ndx = (corr[:, 0] >= 0) & (corr[:, 1] >= 0)

    # use the above generated mask to filter the points in image 1
    x1 = points2D[0][:, corr[ndx, 0]]
    x1 = np.vstack( (x1, np.ones(x1.shape[1])) )

    # use the above generated mask to filter the points in image 2
    x2 = points2D[1][:, corr[ndx, 1]]
    x2 = np.vstack( (x2, np.ones(x2.shape[1])) )


    '''Given n pairs of points, returns their 3d coordinates.'''
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    X = [triangulate_point(x1[:, i], x2[:, i], P1, P2) for i in range(n)]
    return np.array(X).T


def triangulate_point(x1, x2, P1, P2):
  '''Given two image coordinates x1, x2 of the same point X under different
  projections P1, P2, recovers X.'''
  M = np.zeros((6, 6))
  M[:3, :4] = P1
  M[:3, 4] = -x1

  M[3:, :4] = P2
  M[3:, 5] = -x2  # Intentionally 5, not 4.

  U, S, V = np.linalg.svd(M)
  X = V[-1, :4]
  return X / X[3]