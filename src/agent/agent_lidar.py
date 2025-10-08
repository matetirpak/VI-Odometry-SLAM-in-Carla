from matplotlib import cm
import numpy as np
import open3d as o3d
import time

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def process_lidar(self, data):
    start_time = time.time()

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Nx3 LiDAR points
    sensor_points = data.T
    # Convert to camera/SLAM coordinate system
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2]*-1,
        sensor_points[0],
        np.zeros(sensor_points.shape[1])
    ]).T

    # Create Open3D point cloud
    pcd_cpu = o3d.geometry.PointCloud()
    pcd_cpu.points = o3d.utility.Vector3dVector(point_in_camera_coords[:,:3])
    pcd_cpu.colors = o3d.utility.Vector3dVector(int_color)

    # Transform to SLAM
    pcd_cpu.transform(self.RT)

    # Skip ICP for performance. NDT would be better but is not in Open3D
    """
    if len(self.all_points) > 0:
        target_down = self.point_list.voxel_down_sample(voxel_size=0.3)
        target_down.estimate_normals()

        pcd_down = pcd_cpu.voxel_down_sample(voxel_size=0.3)
        pcd_down.estimate_normals()

        threshold = 0.2
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_down, target_down, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
    """

    # Concatenate using numpy
    existing_points = np.asarray(self.point_list.points)
    existing_colors = np.asarray(self.point_list.colors)
    new_points = np.asarray(pcd_cpu.points)
    new_colors = np.asarray(pcd_cpu.colors)
    all_points = np.vstack([existing_points, new_points]) if existing_points.size else new_points
    all_colors = np.vstack([existing_colors, new_colors]) if existing_colors.size else new_colors

    # Update point_list 
    self.point_list.points = o3d.utility.Vector3dVector(all_points)
    self.point_list.colors = o3d.utility.Vector3dVector(all_colors)


def show_lidar_vis(agent):
    if not agent.vis_geometry_initialized:
        agent.vis.add_geometry(agent.point_list)
        agent.vis_geometry_initialized = True

    if len(agent.point_list.points) > 0:
        agent.vis.update_geometry(agent.point_list)

    agent.vis.poll_events()
    agent.vis.update_renderer()


def init_module_variables(agent):
    # Try to enable GPU acceleration
    try:
        o3d.core.Device("cuda:0")
        device = o3d.core.Device("cuda:0")
        print("Open3D GPU acceleration enabled")
    except:
        device = o3d.core.Device("cpu:0")
        print("Using Open3D CPU processing")
    
    agent.vis = o3d.visualization.Visualizer()
    agent.vis.create_window(
        window_name='Carla Lidar',
        width=1280,
        height=720,
        left=480,
        top=270)
    agent.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    agent.vis.get_render_option().point_size = 1
    agent.vis.get_render_option().show_coordinate_frame = True

    agent.lidar_bboxes = o3d.geometry.LineSet()
    agent.vis.add_geometry(agent.lidar_bboxes)
    agent.point_list = o3d.geometry.PointCloud()
    agent.vis_geometry_initialized = False


def add_module_to_agent(agent):
    '''Adds all module functionalities to an object.
    Must be called in __init__.
    '''
    init_module_variables(agent)
    # Functions
    agent.process_lidar = process_lidar.__get__(agent)
    agent.show_lidar_vis = show_lidar_vis.__get__(agent)
