import numpy as np
import cv2
from skimage import transform as tf
import scipy

from helpers import *


def getFeatures(img,bbox):
    """
    Description: Identify feature points within bounding box for each object
    Input:
        img: Grayscale input image, (H, W)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in first frame, (F, N, 2)
    Instruction: Please feel free to use cv2.goodFeaturesToTrack() or cv.cornerHarris()
    """
    bbox = bbox.astype(int)

    features = np.zeros( ( bbox.shape[0], 25, 2 ) )
    for i in range(bbox.shape[0]):
        image_within_bounding_box = img[ bbox[i, 0, 1] : bbox[i, 1, 1] + 1 , bbox[i, 0, 0] : bbox[i, 1, 0] + 1 ]
        corners = cv2.goodFeaturesToTrack( image_within_bounding_box, 25, 0.01, 5 )

        if corners is None:
            features[i] = 0
            continue

        x = corners[:, 0, 0] + bbox[i, 0, 0]
        y = corners[:, 0, 1] + bbox[i, 0, 1]

        features[ i, :corners.shape[0], 0 ] = x
        features[ i, :corners.shape[0], 1 ] = y

    return features

def findGradients(img, ksize=5, sigma=1):
    # Finding the Gaussian kernel (https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa)
    G = cv2.getGaussianKernel( ksize, sigma )
    G = G @ G.T
    
    dx = np.array( [ [1, -1] ] )
    dy = dx.T
    
    Gx = scipy.signal.convolve2d(G, dx, 'same', 'symm')[:, 1:]
    Gy = scipy.signal.convolve2d(G, dy, 'same', 'symm')[1:, :]
    
    Jx = scipy.signal.convolve2d(img, Gx, 'same', 'symm')
    Jy = scipy.signal.convolve2d(img, Gy, 'same', 'symm')

    return Jx, Jy

def estimateFeatureTranslation(feature, Ix, Iy, img1, img2):
    """
    Description: Get corresponding point for one feature point
    Input:
        feature: Coordinate of feature point in first frame, (2,)
        Ix: Gradient along the x direction, (H,W)
        Iy: Gradient along the y direction, (H,W)
        img1: First image frame, (H,W,3)
        img2: Second image frame, (H,W,3)
    Output:
        new_feature: Coordinate of feature point in second frame, (2,)
    Instruction: Please feel free to use interp2() and getWinBound() from helpers
    """
    size_of_window = 15
    win_left, win_right, win_top, win_bottom = getWinBound( img1.shape, feature[0], feature[1], size_of_window )

    windowed_img1 = img1[ win_top : win_bottom, win_left : win_right ]
    windowed_img2 = img2[ win_top : win_bottom, win_left : win_right ]

    du_sum = 0
    dv_sum = 0

    for i in range( 20 ):
        Jx, Jy = findGradients( windowed_img2, 5, 1 )
        Id = windowed_img2 - windowed_img1

        A = np.zeros( ( 2, 2 ) )
        A[ 0, 0 ] = np.sum( np.square( Jx.reshape( -1, 1 ) ) )
        A[ 0, 1] = np.sum( Jx.reshape( -1, 1 ) * Jy.reshape( -1, 1 ) )
        A[ 1, 0] = A[ 0, 1 ].copy()
        A[ 1, 1 ] = np.sum( np.square( Jy.reshape( -1, 1 ) ) )
        
        b = np.zeros( ( 2, 1 ) )
        b[0,0] = -np.sum( Jx.reshape(-1,1) * Id.reshape( -1, 1 ) )
        b[1,0] = -np.sum( Jy.reshape(-1,1) * Id.reshape( -1, 1 ) )

        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
        try:
            result = np.linalg.solve( A, b)
        except np.linalg.LinAlgError as e:
            continue

        du_sum += result[ 0, 0 ]
        dv_sum += result[ 1, 0 ]

        win_left, win_right, win_top, win_bottom = getWinBound( img2.shape, feature[0], feature[1], size_of_window )
        x, y = np.meshgrid( np.arange( win_left, win_right), np.arange( win_top, win_bottom ) )
        
        x = x + du_sum
        y = y + dv_sum

        windowed_img2 = interp2( img2, x, y )

    return np.array( [ feature[0] + du_sum, feature[1] + dv_sum ] )

def estimateAllTranslation(features, img1, img2):
    """
    Description: Get corresponding points for all feature points
    Input:
        features: Coordinates of all feature points in first frame, (F, N, 2)
        img1: First image frame, (H,W,3)
        img2: Second image frame, (H,W,3)
    Output:
        new_features: Coordinates of all feature points in second frame, (F, N, 2)
    """
    new_features = np.zeros( features.shape )
    
    for i in range( features.shape[0] ):
        for j in range( features.shape[1] ):
            
            if features[ i, j, 0] == 0 and features[ i, j, 1 ] == 0: continue

            new_features[ i, j, :] = estimateFeatureTranslation(features[i,j,:], None, None, img1, img2)
     
    return new_features

def applyGeometricTransformation(features, new_features, bbox):
    """
    Description: Transform bounding box corners onto new image frame
    Input:
        features: Coordinates of all feature points in first frame, (F, N, 2)
        new_features: Coordinates of all feature points in second frame, (F, N, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in first frame after eliminating outliers, (F, N1, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Instruction: Please feel free to use skimage.transform.estimate_transform()
    """

    for i in range( features.shape[0] ):    
        non_zero_entries = ~np.logical_and( ( features[ i, :, 0 ]==0 ), ( features[ i, :, 1 ] == 0 ) )
        features_p = features[ i, non_zero_entries, : ]
        new_features_p = new_features [ i, non_zero_entries, : ]
        tform = tf.estimate_transform('similarity', features_p, new_features_p)
        
        bbox[i] = tform( bbox[i] )

    return new_features, bbox

