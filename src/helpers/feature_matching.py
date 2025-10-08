import cv2
import matplotlib.pyplot as plt
import numpy as np


MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def get_image_features(image, n_features=1000):
    """
    Uses ORB to extract image features.
    Returns keypoints and descriptors.
    """

    orb = cv2.ORB_create()
    kp = cv2.goodFeaturesToTrack(
        np.mean(image, axis=2).astype(np.uint8), 
        3000, 
        qualityLevel=0.01, 
        minDistance=7
    )
    kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in kp]
    kp, des = orb.compute(image, kp)
    return kp, des


def match_two_images(des1, des2, threshold=0.75):
    """
    Generates and returns good matches for two images using Lowe's ratio test.
    """

    knn_matches = MATCHER.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m in knn_matches:
        # Only keep matches where the first match is significantly better
        if len(m) == 2 and m[0].distance < threshold * m[1].distance:
            good_matches.append(m[0])

    return good_matches


def estimate_movement(matches, kp1, kp2, K, depth_map=None):
    if depth_map is not None:
        image2_points = [] # Points on the second image
        object_points = [] # Same points in 3D space

        # Build corresponding 2D-3D points iteratively
        for m in matches:
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt
            z = depth_map[int(round(v1)), int(round(u1))]
            
            if z >= 1000:
                continue
            p_cam = np.linalg.inv(K) @ (z * np.array([u1,v1,1]))
            
            image2_points.append([u2,v2])
            object_points.append(p_cam)
        
        image2_points = np.array(image2_points)
        object_points = np.vstack(object_points)
        
        # Estimate pose / rotation and translation
        _, rvec, tvec, _ = cv2.solvePnPRansac(
            object_points, 
            image2_points, 
            K, 
            None
        )
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
    
    else:
        # Use essential matrix method if no depth map is provided
        image1_points = []
        image2_points = []
        for m in matches:
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt
            image1_points.append([u1,v1])
            image2_points.append([u2,v2])

        # Estimate pose
        E, _ = cv2.findEssentialMat(
            np.array(image1_points), 
            np.array(image2_points), 
            K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )

        # Recover pose from essential matrix
        _, rmat, tvec, _ = cv2.recoverPose(
            E, 
            np.array(image1_points), 
            np.array(image2_points), 
            K
        )
    
    return rmat, tvec


##################
# Visualizations #
##################
def add_keypoints(image, kp):
    """
    Given an image and keypoints, returns the image with keypoints drawn.
    """

    image_output = cv2.drawKeypoints(image, kp, None, color=(255, 255, 255),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    return image_output