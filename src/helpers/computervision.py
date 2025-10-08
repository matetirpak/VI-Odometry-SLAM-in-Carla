import numpy as np

def build_intrinsic_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    """
    Calculates the 2D image point from a 3D world location.
    """
    point = np.array([loc.x, loc.y, loc.z, 1])
    
    # Transform to camera coordinates
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # Project 3D->2D using the intrinsic camera matrix
    point_img = np.dot(K, point_camera)
    
    # Normalize by depth
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[:2]

def point_in_canvas(pos, img_h, img_w):
    """
    Returns true if point is in canvas.
    """
    if (pos[0] >= 0) and (pos[0] < img_w) and \
    (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

def xy_from_depth(depth, k):
    """
    Given a depth map and the intrinsic camera matrix, computes the x,
    and y coordinates of every pixel in the image.
    Inputs and outputs are numpy arrays of shape (H, W).
    """

    h, w = depth.shape
    c_u, c_v = k[0, 2], k[1, 2]
    f = k[0, 0]

    # Shape (H, W)
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))  

    x = (u_coords - c_u) * depth / f
    y = (v_coords - c_v) * depth / f
    
    return x, y