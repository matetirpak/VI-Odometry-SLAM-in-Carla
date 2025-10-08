import argparse
import carla
import copy
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
import random
import time
import queue
import warnings

from src.helpers.computervision import *

import src.agent.agent_gt_boxes as agent_gt_boxes
import src.agent.agent_imu_odometry as agent_imu_odometry
import src.agent.agent_lidar as agent_lidar
import src.agent.agent_odometry as agent_odometry
import src.agent.agent_visual_odometry as agent_visual_odometry
#import src.agent.agent_kalman_filter as agent_kalman_filter
import src.agent.agent_video_recorder as agent_video_recorder

class Agent:
    def __init__(self, fps, odometry_mode, show_camera, show_depth_camera,
    show_trajectory, enable_slam, draw_camera_gt_boxes, draw_lidar_gt_boxes,
    n_cars, record_video=False, video_output='slam_demo.mp4'):
        self.show_camera = show_camera
        self.show_depth_camera = show_depth_camera
        self.show_trajectory = show_trajectory
        self.enable_slam = enable_slam
        self.use_visual_odometry = odometry_mode == 'visual'
        self.use_inertial_odometry = odometry_mode == 'inertial'
        self.draw_camera_gt_boxes = draw_camera_gt_boxes != 'none'
        self.draw_camera_gt_boxes_3d = draw_camera_gt_boxes == '3d'
        self.draw_lidar_gt_boxes = draw_lidar_gt_boxes

        self.record_video = record_video
        self.video_output = video_output

        self.max_world_fps = fps
        self.delta_seconds = 1 / self.max_world_fps
        
        # Processing every dt frames
        self.camera_dt = math.ceil(fps * 0.1)
        self.imu_dt = 1
        self.lidar_dt = math.ceil(fps * 0.2)

        self.world_frame = -1 # Arbitrary value to avoid "not initialized" errors
        self.rgb_frame_world_tick = -1
        self.depth_map_world_tick = -1
        self.imu_data_world_tick = -1
        self.lidar_data_world_tick = -1

        # Variables for sensor data
        self.rgb_frame = None
        self.depth_map = None
        self.imu_data = None
        self.lidar_data = None

        print("Connecting to Carla server...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        print("Connected to Carla server.")
        self.world = self.client.get_world()

        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=70.0,
            fog_density=0.0,
            wetness=0.0
        )
        self.world.set_weather(weather)
        self.world.apply_settings(
            carla.WorldSettings(
                no_rendering_mode=True,
                synchronous_mode=True,
                fixed_delta_seconds=self.delta_seconds
            )
        )

        bp_lib = self.world.get_blueprint_library() 
        
        # Spawn ego vehicle at a random available spot
        vehicle_bp = bp_lib.find('vehicle.audi.etron')
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        ego_vehicle = None
        for i in range(len(spawn_points)):
            ego_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[i])
            if ego_vehicle:
                print(f"Spawned ego vehicle at spawn point {i}")
                break
        if ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle at any spawn point.")
        ego_vehicle.set_autopilot(True)
        self.ego_vehicle = ego_vehicle

        # Spawn npc cars
        self.car_actors = []
        for i in range(n_cars):  
            vehicle_bp = random.choice(bp_lib.filter('vehicle'))
            # Use navigation location to ensure spawning on ground
            spawn_points = self.world.get_map().get_spawn_points()
            if spawn_points:
                spawn_transform = random.choice(spawn_points)
                npc = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
                if npc is not None:
                    self.car_actors.append(npc)
                    npc.set_autopilot(True)
                    print(f"Spawned NPC car {i+1}/{n_cars}")
                else:
                    print(f"Failed to spawn NPC car {i+1}")
            else:
                print(f"No navigation location available for NPC car {i+1}")
                print(f"No navigation location available for NPC car {i+1}")
        
        # Camera parameters
        self.image_w = 1024
        self.image_h = 512
        self.fov = 90
        self.K = build_intrinsic_matrix(self.image_w, self.image_h, self.fov)
        self.K_b = build_intrinsic_matrix(
            self.image_w,
            self.image_h,
            self.fov,
            is_behind_camera=True
        )

        self.sensors = []

        # Camera sensor
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", str(self.image_w))
        camera_bp.set_attribute("image_size_y", str(self.image_h))
        camera_bp.set_attribute("fov", str(self.fov))
        camera_init_trans = carla.Transform(carla.Location(x=-0.1,z=1.7)) 
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_init_trans,
            attach_to=ego_vehicle
        )
        self.sensors.append(self.camera)

        # Camera depth sensor
        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(self.image_w))
        depth_bp.set_attribute('image_size_y', str(self.image_h))
        depth_bp.set_attribute("fov", str(self.fov))
        depth_init_trans = carla.Transform(carla.Location(x=-0.1, z=1.7))
        self.depth_camera = self.world.spawn_actor(
            depth_bp,
            depth_init_trans,
            attach_to=ego_vehicle
        )
        self.sensors.append(self.depth_camera)

        # IMU sensor
        imu_bp = bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', str(self.delta_seconds))
        imu_bp.set_attribute('noise_accel_stddev_x', '0.2')
        imu_bp.set_attribute('noise_gyro_stddev_z', '0.02')
        imu_init_trans = carla.Transform(carla.Location(x=-0.1, z=1.7))
        self.imu_sensor = self.world.spawn_actor(
            imu_bp,
            imu_init_trans,
            attach_to=ego_vehicle
        )
        self.sensors.append(self.imu_sensor)

        if self.enable_slam:
            # Lidar sensor
            lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('rotation_frequency', str(self.max_world_fps))
            lidar_bp.set_attribute('channels', '128')
            lidar_bp.set_attribute('points_per_second', '500000')
            lidar_bp.set_attribute('upper_fov', '30')
            lidar_bp.set_attribute('lower_fov', '-25')
            lidar_init_trans = carla.Transform(carla.Location(x=0, z=1.7))
            self.lidar_sensor = self.world.spawn_actor(
                lidar_bp,
                lidar_init_trans,
                attach_to=ego_vehicle
            )
            self.sensors.append(self.lidar_sensor)
            self.lidar_sensor.listen(lambda data: self.lidar_callback(data))


        # Collects sensor data to be processed each frame
        self.sensor_data_queue = queue.Queue()

        # Initialize the world
        self.tick_n = 0
        self.world.tick()

        self.n_sensors = len(self.sensors)
        self.camera.listen(lambda img: self.camera_callback(img, 'rgb'))
        self.depth_camera.listen(lambda img: self.camera_callback(img, 'depth'))
        self.imu_sensor.listen(lambda data: self.imu_callback(data))

        # Initialize specific functionalities separately
        agent_visual_odometry.add_module_to_agent(self)
        agent_gt_boxes.add_module_to_agent(self)
        agent_odometry.add_module_to_agent(self)
        agent_imu_odometry.add_module_to_agent(self)
        #agent_kalman_filter.add_module_to_agent(self)
        if self.enable_slam:
            agent_lidar.add_module_to_agent(self)
        if self.record_video:
            agent_video_recorder.add_module_to_agent(self)

    def run(self):
        """
        Main loop.
        """
        
        assert self.world.get_settings().synchronous_mode

        for _ in range(3): # For some reason IMU yields garbage values at the start
            self.world.tick()
        if self.show_trajectory:
            self.init_odometry_plot()

        self.ego_vehicle.set_autopilot(True)
        trajectory_updated = False
        while True:
            start_time = time.perf_counter()

            self.update_world()

            # Inertial odometry (imu)
            if self.imu_data_world_tick == self.tick_n:
                if self.use_inertial_odometry:
                    self.update_imu_state(self.imu_data, self.delta_seconds)
                    self.step_io_trajectory(self.imu_data, self.delta_seconds)
                    self.step_gt_trajectory()
                    trajectory_updated = True

            # Visual odometry (feature matching)
            if self.rgb_frame_world_tick == self.tick_n:
                if self.use_visual_odometry:
                    self.step_vo_trajectory(self.rgb_frame)
                    self.step_gt_trajectory()
                    trajectory_updated = True
                if self.draw_camera_gt_boxes:
                    self.rgb_frame = self.draw_boxes(
                        self.rgb_frame,
                        three_d=self.draw_camera_gt_boxes_3d
                    )

            # Lidar
            if self.lidar_data_world_tick == self.tick_n:
                if self.enable_slam:
                    self.process_lidar(self.lidar_data)

            if self.enable_slam:
                self.show_lidar_vis()
                if self.draw_lidar_gt_boxes:
                    self.draw_lidar_boxes()

            # Draw trajectory every 5 frames
            if self.show_trajectory and self.tick_n % 5 == 0:
                if trajectory_updated:
                    # Only plot if an update happened
                    trajectory_updated = False
                    
                    # Plot
                    estimated_trajectory = self.vo_trajectory if self.use_visual_odometry else self.io_trajectory
                    figname = "trajectory_vo" if self.use_visual_odometry else "trajectory_io"
                    self.draw_trajectory(
                        estimated=estimated_trajectory,
                        actual=self.gt_trajectory,
                        figname=figname
                    )

            # Capture video frame every iteration
            if self.record_video:
                self.capture_video_frame()

            # Display camera images
            if self.show_camera and self.rgb_frame is not None:
                cv2.imshow(
                    "RGB_Image",
                    cv2.cvtColor(self.rgb_frame, cv2.COLOR_BGR2RGB)
                )
            if self.show_depth_camera and self.depth_map is not None:
                cv2.imshow(
                    "Depth",
                    (np.clip(self.depth_map, 0, 100) / 100 * 255).astype(np.uint8)
                )
            
            if cv2.waitKey(1) == ord('q'):
                self.shutdown()

            # Sleep to match real-time FPS
            elapsed = time.perf_counter() - start_time
            sleep_duration = max(0, self.delta_seconds - elapsed)
            time.sleep(sleep_duration)

            self.tick_n += 1

    def update_world(self):
        """
        Advances the world by one step and collects sensor data.
        """
        start_time = time.perf_counter()
        self.world.tick()
        self.world_frame = self.world.get_snapshot().frame
        try:
            for i in range(self.n_sensors):
                data, name = self.sensor_data_queue.get(timeout=0.05)
                
                if name == 'rgb' and self.tick_n % self.camera_dt == 0:
                    self.rgb_frame = data
                    self.rgb_frame_world_tick = self.tick_n
                
                if name == 'depth' and self.tick_n % self.camera_dt == 0:
                    self.depth_map = self.decode_depth_image(data)
                    self.depth_map_world_tick = self.tick_n
                
                if name == 'imu' and self.tick_n % self.imu_dt == 0:
                    self.imu_data = data
                    self.imu_data_world_tick = self.tick_n
                
                if name == 'lidar' and self.tick_n % self.lidar_dt == 0:
                    self.lidar_data = data
                    self.lidar_data_world_tick = self.tick_n
        except queue.Empty:
            msg = f"Only received {i}/{self.n_sensors} sensor outputs in time."
            warnings.warn(msg)

    def lidar_callback(self, point_cloud):
        """
        Callback for lidar sensor. Formats point cloud into [N, 4] numpy array.
        """
        if point_cloud.frame != self.world_frame:
            msg = "Ignoring lidar sample that doesn't match the world timestamp."
            warnings.warn(msg)
            return
            
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        self.sensor_data_queue.put((data, 'lidar'))

    def imu_callback(self, imu_data):
        """
        Callback for IMU sensor.
        """
        
        if imu_data.frame != self.world_frame:
            msg = "Ignoring imu sample that doesn't match the world timestamp."
            warnings.warn(msg)
            return
        
        self.sensor_data_queue.put((imu_data, 'imu'))

    def camera_callback(self, image, name):
        """
        Callback for camera sensors. Converts to RGB numpy array.
        """
        
        if image.frame != self.world_frame:
            msg = "Ignoring camera frame that doesn't match the world timestamp."
            warnings.warn(msg)
            return
        
        img_bgra = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        ).reshape((image.height, image.width, 4))
        img_rgb = np.ascontiguousarray(img_bgra[:, :, [2, 1, 0]])
        self.sensor_data_queue.put((img_rgb, name))

    def decode_depth_image(self, image_rgb):
        """
        Decode depth image from carla RGB format to depth in meters.
        """
        
        img = image_rgb.astype(np.uint32)

        # r + g*256 + b*256^2
        normalized = (img[:, :, 0] + img[:, :, 1] * 256 + img[:, :, 2] * 256 * 256) / (256 ** 3 - 1)

        # Convert to depth in meters (CARLA encodes 0 to 1000m)
        depth_meters = 1000.0 * normalized
        
        return depth_meters
    
    def shutdown(self):
        cv2.destroyAllWindows()
        
        # Close video writer if recording
        if self.record_video and self.video_writer is not None:
            self.video_writer.release()
            print(f"Video saved to: {self.video_output}")
        
        # Destroy sensors
        for sensor in self.sensors:
            try:
                sensor.destroy()
            except Exception as e:
                print(f"Failed to destroy sensor {sensor.type_id}: {e}")
        
        # Destroy vehicles
        for actor in self.car_actors:
            try:
                actor.destroy()
            except Exception as e:
                print(f"Error destroying actor {actor.id}: {e}")
        try:
            self.ego_vehicle.destroy()
        except Exception as e:
            print(f"Error destroying ego vehicle: {e}")

        # Unnecessary
        self.car_actors = []
        self.sensors = []


