import numpy as np
import cv2
import os
import tempfile

def capture_video_frame(self):
    """
    Capture frames from all visualizations and composes
    them into a single video frame.
    """
    if not self.record_video or self.video_writer is None:
        return
        
    frames = []
    
    # Capture RGB camera frame
    if self.show_camera and self.rgb_frame is not None:
        try:
            # Convert RGB to BGR for OpenCV
            camera_frame = cv2.cvtColor(self.rgb_frame, cv2.COLOR_RGB2BGR)
            # Keep original resolution for better quality
            camera_frame = cv2.resize(
                camera_frame,
                (self.image_w, self.image_h)
            )
            frames.append(('camera', camera_frame))
        except Exception as e:
            print(f"Failed to capture camera frame: {e}")

    # Capture depth camera frame
    if self.show_depth_camera and self.depth_map is not None:
        try:
            depth_frame_clipped = np.clip(self.depth_map, 0, 100)
            depth_frame = (depth_frame_clipped / 100 * 255).astype(np.uint8)
            depth_frame = cv2.resize(
                depth_frame,
                (self.image_w, self.image_h)
            )
            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
            frames.append(('depth', depth_frame))
        except Exception as e:
            print(f"Failed to capture depth frame: {e}")

    # Capture trajectory plot (matplotlib)
    if self.show_trajectory and hasattr(self, 'fig'):
        try:
            # Get the current matplotlib figure
            self.fig.canvas.draw()
            width, height = self.fig.get_size_inches() * self.fig.dpi
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(int(height), int(width), 3)
            
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Keep original high resolution for better quality
            # Original is ~1500x400 at 100 DPI, we'll keep it close to that
            img = cv2.resize(img, (int(width), int(height)))  # No resize, keep original
            frames.append(('trajectory', img))
        except Exception as e:
            print(f"Failed to capture trajectory plot: {e}")
    
    # Capture LiDAR point cloud visualization (Open3D)
    if self.enable_slam and hasattr(self, 'vis'):
        try:
            # Create a temporary file for the screenshot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Capture screenshot from Open3D visualizer
            self.vis.capture_screen_image(temp_filename)
            
            # Read the image with OpenCV
            lidar_frame = cv2.imread(temp_filename)
            if lidar_frame is not None:
                # Resize to match camera dimensions for consistency
                frames.append(('lidar', lidar_frame))
            
            # Clean up temporary file
            os.unlink(temp_filename)
        except Exception as e:
            print(f"Failed to capture LiDAR visualization: {e}")
    
    # Compose frames into a single video frame
    if frames:
        self.compose_and_write_video_frame(frames)

def compose_and_write_video_frame(self, frames):
    """
    Compose multiple frames into a single video frame and writes it to video.
    """
    canvas = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
    
    if len(frames) == 1:
        # Single frame - full canvas
        frame = frames[0][1]
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        canvas = frame
    elif len(frames) == 2:
        # Two frames - side by side
        frame1 = frames[0][1]
        frame2 = frames[1][1]
        
        # Calculate target dimensions to fit both frames
        target_width = self.video_width // 2
        target_height = self.video_height
        
        frame1 = cv2.resize(frame1, (target_width, target_height))
        frame2 = cv2.resize(frame2, (target_width, target_height))
        
        canvas[:, :target_width] = frame1
        canvas[:, target_width:] = frame2
    elif len(frames) == 3:
        # Three frames - 1 full width, 2 half width
        trajectory_frame = None
        other_frames = []
        
        for name, frame in frames:
            if name == 'trajectory':
                trajectory_frame = frame
            else:
                other_frames.append(frame)
        
        if trajectory_frame is not None and len(other_frames) == 2:
            # Top section for trajectory (full width)
            traj_height = self.video_height // 2
            trajectory_resized = cv2.resize(
                trajectory_frame,
                (self.video_width, traj_height)
            )
            canvas[:traj_height, :] = trajectory_resized
            
            # Bottom section split for other two frames
            bottom_height = self.video_height - traj_height
            frame1 = cv2.resize(
                other_frames[0],
                (self.video_width // 2, bottom_height)
            )
            frame2 = cv2.resize(
                other_frames[1],
                (self.video_width // 2,
                bottom_height)
            )
            
            canvas[traj_height:, :self.video_width//2] = frame1
            canvas[traj_height:, self.video_width//2:] = frame2
        else:
            positions = [
                (0, 0),
                (0, self.video_width//2),
                (self.video_height//2, self.video_width//4)
            ]
            for i, (name, frame) in enumerate(frames):
                if i < 3:
                    y, x = positions[i]
                    if i == 2:  # Third frame gets full bottom width
                        resized_frame = cv2.resize(
                            frame,
                            (self.video_width, self.video_height//2)
                        )
                        canvas[y:, :] = resized_frame
                    else:
                        resized_frame = cv2.resize(
                            frame,
                            (self.video_width//2, self.video_height//2)
                        )
                        y_to = y+self.video_height//2
                        x_to = x+self.video_width//2
                        canvas[y:y_to, x:x_to] = resized_frame
    elif len(frames) == 4:
        # Four frames - 2x2 grid
        positions = [
            (0, 0),  # Top-left
            (0, self.video_width//2),  # Top-right  
            (self.video_height//2, 0),  # Bottom-left
            (self.video_height//2, self.video_width//2)  # Bottom-right
        ]
        
        for i, (name, frame) in enumerate(frames[:4]):
            y, x = positions[i]
            resized_frame = cv2.resize(
                frame,
                (self.video_width//2, self.video_height//2)
            )
            y_to = y+self.video_height//2
            x_to = x+self.video_width//2
            canvas[y:y_to, x:x_to] = resized_frame
    else:
        msg = f"Unsupported number {len(frames)} of frames for composition."
        raise ValueError(msg)

    
    # Add timestamp/frame number
    cv2.putText(
        canvas,
        f"Frame: {self.tick_n}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.5, 
        (255, 255, 255),
        3
    )
    
    # Write frame to video
    self.video_writer.write(canvas)

def init_module_variables(agent):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    agent.video_width = 2560
    agent.video_height = 1440
    video_fps = agent.max_world_fps
    agent.video_writer = cv2.VideoWriter(agent.video_output, fourcc, video_fps, (agent.video_width, agent.video_height))
    print(f"Video recording enabled. Output: {agent.video_output} at {video_fps} FPS ({agent.video_width}x{agent.video_height})")
    print(f"Capturing video frames every world tick for maximum responsiveness")

def add_module_to_agent(agent):
    '''Adds all module functionalities to an object.
    Must be called in __init__.
    '''
    init_module_variables(agent)
    # Functions
    agent.compose_and_write_video_frame = compose_and_write_video_frame.__get__(agent)
    agent.capture_video_frame = capture_video_frame.__get__(agent)