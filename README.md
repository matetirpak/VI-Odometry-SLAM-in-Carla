## Results
![](demos/slam_demo_1_540p.gif)
The GIF demonstrates the real-time performance of the SLAM system, with the estimated trajectory plotted in the bottom-left panel and the accumulated LiDAR map in the bottom-right. The trajectory estimation is highly accurate throughout most of the sequence, showcasing the effectiveness of Visual Odometry despite its relative simplicity compared to more complex algorithms. Minor drift accumulates toward the end, which would be more apparent in scenarios involving loop closures, but the overall map quality remains strong.

## Setup
Make sure both Carla and a python environment are set up. 
Python 3.7.17 was used for testing.
```bash
git clone https://github.com/matetirpak/VI-Odometry-SLAM-in-Carla.git
cd VI-Odometry-SLAM-in-Carla
pip install -r requirements.txt
# Start CarlaUE4.sh
# View args with: python3 src/main/main.py -h
python3 src/main/main.py --show_all
```

## Methods
This project was developed to experiment with various odometry methods and gain insight into their quality via SLAM. For visualization, LiDAR-generated point clouds are aligned using the odometry data. Visual Odometry offers the best trade-off between reliability and simplicity. IMU-based Inertial Odometry is prone to drift and requires a very fast measurement delta which doesn't integrate well with the simulator loop of the current implementation. Although more reliable than Inertial Odometry, using Visual Odometry alone is insufficient for real-world scenarios as errors in the estimation never get corrected and accumulate. Methods to address these issues and possible expansions will be discussed shortly.

### Visual Odometry
Estimating motion using camera data requires estimating the poses of pairs of consecutive images. To advance the global estimate at frame $\mathbf{t}$, the local motion between frame $\mathbf{t-1}$ and $\mathbf{t}$ has to be estimated and then applied globally. The following process describes the motion estimation between two frames.

First, features and descriptors are generated for both images using ORB, an efficient and rotation-invariant feature extractor. In a circular shape, it searches for points with nearby local pixel differences. Compared to SIFT and SURF, it is computationally more efficient, doesn't have a noticeable downside in quality and is patent-free, making it a great choice.

Having extracted the features, each of them is matched with the features of the other image, generating at most as many matches as features. Applying Lowe's ratio test helps discard matches of low quality. Only those are kept, which have a large enough difference to the second best match of the respective feature, introducing an important confidence threshold. 

Precisely, keep a match between feature $\mathbf{f}_i$ and its best match $\mathbf{f}_j$ only if

$$
\frac{\text{distance}(\mathbf{f}_i, \mathbf{f}_j^{(1)})}{\text{distance}(\mathbf{f}_i, \mathbf{f}_j^{(2)})} < \mathbf{\tau}
$$



Features are extracted and matched. What's left is estimating the transformation between the two images. Assuming depth is given, either through a depth camera or stereo vision, and the intrinsic camera matrix $\mathbf{K}$ is known, estimating the poses of the two frames can be solved in the 3D space by computing the 3D coordinates at each pixel $\mathbf{p_i} = (u_i, v_i)$ of the first frame and given depth $d_i$ using the formula 

$$\mathbf{P}_i = \mathbf{d}_i \, \mathbf{K}^{-1} \begin{bmatrix} u_i \\\\ v_i \\\\ 1 \end{bmatrix}$$

and then applying the Perspective-n-Point (PnP) algorithm. PnP solves for the camera's extrinsic parameters, rotation matrix $\mathbf{R}$ and translation vector $\mathbf{t}$, by minimizing the reprojection error between observed 2D image points $\mathbf{p_i} = (u_i, v_i)$ of the second image and their projections from known 3D world points $\mathbf{P}_i = (X_i, Y_i, Z_i)$ of the first image. Intuitively, PnP computes a mapping from the latest world state to a new incoming image. The projection model is given by 

$$p_i \approx \mathbf{K} (\mathbf{R} \mathbf{P}_i + \mathbf{t})$$

Using nonlinear least-squares optimization, PnP iteratively refines $\mathbf{R}$ and $\mathbf{t}$ by minimizing the objective function 

$$\min_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^{N} \lVert  \mathbf{p}_i - \mathbf{\hat{p}}_i \\ \rVert^2$$

to best align the 3D-to-2D mappings. Both rotation and translation have three degrees of freedom, resulting in PnP requiring at least three 2D points (and their 3D correspondences) to match the six degrees with six equations and an additional fourth point to avoid potential ambiguity. However, in the case of feature matching, this is easily satisfied and not of concern.

Working with such a low parameter count is prone to inconsistencies and noise. To stabilize the process, Random Sample Consensus \(RANSAC\) is used to reject such outliers. Iteratively, a subset of point correspondences is chosen to perform PnP on. The resulting transformation is then evaluated by counting the number of point correspondences that agree with it within a certain error threshold, referred to as inliers. After the iterations conclude or are stopped early upon reaching a certain number of inliers, the transformation is retained and optionally refined by recomputing PnP using all inliers.

The result is a matrix $\mathbf{RT}\_{\text{local}}$ representing the relative transformation (rotation and translation) between the previous and current frame. Multiplying the transformation matrix up to the previous frame $\mathbf{RT}\_{\text{global}}$, with the local $\mathbf{RT}\_{{\text{local}}}^{-1}$ advances the global estimate $\mathbf{RT}\_{\text{global}}$ to the current frame.

$$
\mathbf{RT} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\\\ \mathbf{0}^{\top} & 1 \end{bmatrix}, \quad \mathbf{RT}_{\text{new global}} = \mathbf{RT}_{\text{global}} \cdot \mathbf{RT}_{\text{local}}^{-1}
$$

At initialization, $\mathbf{RT}$ is set to the identity matrix $\mathbf{I}$ (no rotation or translation). 

This concludes one step of the Visual Odometry pipeline, where $\mathbf{RT}$ serves as the transformation matrix mapping world points into the estimated camera frame.

$$
\mathbf{RT}_{global} \cdot \begin{bmatrix} x \\\\ y \\\\ z \\\\ 1 \end{bmatrix} = \begin{bmatrix} \hat{x} \\\\ \hat{y} \\\\ \hat{z} \\\\ 1\end{bmatrix}
$$


### LiDAR Mapping
Since this project originated from Visual Odometry, the map has since been based on the conventional camera coordinate system. This is not ideal and will most likely be changed in the future. LiDAR measurements of the current timestep are first converted into the camera coordinate system and then transformed by the estimated global rotation and translation $\mathbf{RT}_\text{{global}}$. It is then sufficient to simply add the transformed current point cloud to the collection of all previous point clouds. Following this procedure iteratively generates a LiDAR map in real time.


## Future Work
A Kalman Filter could be used to stabilize both Inertial and Visual Odometry individually or fuse them together, predicting future motion based on IMU measurements and updating those predictions with visual estimates. In particular, an Extended Kalman Filter (EKF) is recommended, as it effectively handles the nonlinear dynamics of camera and IMU data, overcoming the limitations of loose coupling. These enhancements would directly improve map quality by reducing drift and integrating complementary sensor information.

For additional robustness, postprocessing the estimates with point cloud registration algorithms like Iterative Closest Point (ICP) or Normal Distributions Transform (NDT) could be beneficial. NDT is particularly powerful for localization in known high-definition maps and can perform odometry standalone, while ICP is more popular but prone to local minima in dense point clouds. NDT could be integrated into a Visual-Inertial Odometry setup, serving as a third sensor for fusion, further improving the robustness and trajectory estimation accuracy. However, even with these, drift and local errors may still accumulate over time.

Pose-graph optimization addresses this problem by incorporating loop closures, globally refining the map as revisited locations are recognized. This enables continuous improvement over time and is arguably the strongest approach for long-term SLAM.

In summary, this project highlights the exciting potential of visual odometry and LiDAR mapping in a simulated environment, laying groundwork for more robust SLAM systems. Future extensions, including sensor fusion and global optimization, could bridge the gap to near real-world autonomy. 


## References

[ORB: An efficient alternative to SIFT or SURF - Ethan Rublee; Vincent Rabaud; Kurt Konolige; Gary Bradski](https://ieeexplore.ieee.org/document/6126544)

["Self-Driving Cars Specialization" by the University of Toronto taught on coursera](https://www.coursera.org/specializations/self-driving-cars)

[Official documentation and code examples of Carla](https://carla.readthedocs.io/en/latest/)

[Dynamic plotting functions](https://github.com/AhmedHisham1/carla-visual-odometry)