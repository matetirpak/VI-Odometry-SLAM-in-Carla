import argparse

from src.agent.agent import Agent

def parse_args(parser):
    parser.add_argument('--fps', type=int, default=10,
                        help='Simulation frames per second (default: 10)')

    parser.add_argument('--n_cars', type=int, default=32,
                        help='Amount of npc cars that should be spawned (default: 32)')

    parser.add_argument('--odometry_mode', type=str, choices=['visual', 'inertial'], default='visual',
                        help='Odometry method to use (default: visual). Options: "visual", "inertial"')

    parser.add_argument('--slam', action='store_true',
                        help='Generate and display the LiDAR map. (default: False)')

    parser.add_argument('--show_camera', action='store_true',
                        help='Display the RGB camera view (default: False)')

    parser.add_argument('--draw_camera_gt_boxes', choices=['2d', '3d'], default='none',
                        help='Draw ground-truth bounding boxes in the RGB camera view (default: none)')

    parser.add_argument('--draw_lidar_gt_boxes', action='store_true',
                        help='Draw ground-truth bounding boxes in LiDAR view (default: False)')

    parser.add_argument('--show_depth_camera', action='store_true',
                        help='Display the depth camera view (default: False)')

    parser.add_argument('--show_trajectory', action='store_true',
                        help='Display the odometry trajectory (default: False)')

    parser.add_argument('--show_all', action='store_true',
                        help='Enable all visualizations and options (overrides individual flags)')
    
    parser.add_argument('--record_video', action='store_true',
                        help='Record a video of all visualizations (default: False)')

    parser.add_argument('--video_output', type=str, default='slam_demo.mp4',
                        help='Output video filename (default: slam_demo.mp4)')

    return parser.parse_args()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Configure the CARLA Lidar + Camera visual SLAM pipeline"
    )
    args = parse_args(parser)

    SHOW_ALL = args.show_all
    FPS = args.fps or SHOW_ALL
    N_CARS = args.n_cars
    ODOMETRY_MODE = args.odometry_mode
    ENABLE_SLAM = args.slam or SHOW_ALL
    DRAW_LIDAR_GT_BOXES = args.draw_lidar_gt_boxes or SHOW_ALL
    SHOW_CAMERA = args.show_camera or SHOW_ALL
    DRAW_CAMERA_GT_BOXES = args.draw_camera_gt_boxes or SHOW_ALL
    SHOW_DEPTH_CAMERA = args.show_depth_camera or SHOW_ALL
    SHOW_TRAJECTORY = args.show_trajectory or SHOW_ALL

    RECORD_VIDEO = args.record_video
    VIDEO_OUTPUT = args.video_output

    agent = Agent(
        fps=FPS, 
        odometry_mode=ODOMETRY_MODE,
        show_camera=SHOW_CAMERA, 
        show_depth_camera=SHOW_DEPTH_CAMERA, 
        show_trajectory=SHOW_TRAJECTORY, 
        enable_slam=ENABLE_SLAM, 
        draw_camera_gt_boxes=DRAW_CAMERA_GT_BOXES, 
        draw_lidar_gt_boxes=DRAW_LIDAR_GT_BOXES,
        n_cars=N_CARS,
        record_video=RECORD_VIDEO,
        video_output=VIDEO_OUTPUT
    )
    try:
        agent.run()
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        print("Shutting agent down...")
        agent.shutdown()
    print("Program exited.")
