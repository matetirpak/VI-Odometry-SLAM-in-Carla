import cv2
import numpy as np
import open3d as o3d
import warnings

from src.helpers.computervision import *
from src.agent.agent_detection import min_distance_in_bbox

def get_boxes(agent, only_front=True):
        """
        Collects all bounding boxes of vehicles in the scene.
        Returns:
            List[List[carla.libcarla.BoundingBox, carla.libcarla.Transform]]: [[bb, npc_transform], ...]
        """
        ret_bboxes = []

        ego_location = agent.ego_vehicle.get_transform().location
        ego_forward_vec = agent.ego_vehicle.get_transform().get_forward_vector()
        for npc in agent.world.get_actors().filter('*vehicle*'):
            if npc.id == agent.ego_vehicle.id:
                continue

            bb = npc.bounding_box
            
            # Filter for the vehicles within 50m
            dist = npc.get_transform().location.distance(
                agent.ego_vehicle.get_transform().location
            )
            if dist >= 50:
                continue
            # Filter for vehicles in front of the camera
            if only_front:                 
                ray = npc.get_transform().location - ego_location
                if ego_forward_vec.dot(ray) <= 0:
                    continue
            ret_bboxes.append([bb, npc.get_transform()])
        return ret_bboxes

def draw_lidar_boxes(agent):
    edges = [
        [0,1], [1,3], [3,2], [2,0], [0,4], [4,5],
        [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]
    ]
    # Visualization setup
    if not hasattr(agent, 'vis'):
        raise ValueError("Open3D visualizer not initialized in agent.")
    all_points = []
    all_lines = []
    all_colors = []
    
    bboxes = agent.get_boxes(only_front=False)
    w2c = np.array(agent.camera.get_transform().get_inverse_matrix())
    def process_point(point, w2c):
        point_np = np.array([point.x, point.y, point.z, 1])
        
        # world_position -> camera_position -> camera_coordinate_frame -> slam_position
        point_lidar = agent.RT @ agent.left_to_right_hand_coord_system @ w2c @ point_np

        point_lidar = point_lidar.T[:3]
        return point_lidar
    
    # Save points and connections for each edge of each bounding box
    for bb, npc_transform in bboxes:
        verts = [v for v in bb.get_world_vertices(npc_transform)]
        for edge in edges:
            point1 = process_point(verts[edge[0]], w2c)
            point2 = process_point(verts[edge[1]], w2c)
            
            start_idx = len(all_points)
            all_points.extend([point1, point2])
            all_lines.append([start_idx, start_idx + 1])
            all_colors.append([1.0, 0.0, 0.0])  # red

    # Update visualization
    if len(bboxes) > 0:
        agent.lidar_bboxes.points = o3d.utility.Vector3dVector(
            np.array(all_points))
        agent.lidar_bboxes.lines = o3d.utility.Vector2iVector(
            np.array(all_lines))
        agent.lidar_bboxes.colors = o3d.utility.Vector3dVector(
            np.array(all_colors))

        agent.vis.update_geometry(agent.lidar_bboxes)


def draw_boxes(self, img, min_dist=True, three_d=False):
    edges = [
        [0,1], [1,3], [3,2], [2,0], [0,4], [4,5],
        [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]
    ]

    if min_dist and three_d:
        warnings.warn("Min distance with 3D boxes not implemented yet.")
        min_dist = False
    if min_dist:
        z = self.depth_map
        x, y = xy_from_depth(z, self.K)
    
    world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())
    
    
    bboxes = self.get_boxes()
    for bb, npc_transform in bboxes:
        if three_d:
            verts = [v for v in bb.get_world_vertices(npc_transform)]
            for edge in edges:
                p1 = get_image_point(verts[edge[0]], self.K, world_2_camera)
                p2 = get_image_point(verts[edge[1]],  self.K, world_2_camera)

                p1_in_canvas = point_in_canvas(p1, self.image_h, self.image_w)
                p2_in_canvas = point_in_canvas(p2, self.image_h, self.image_w)

                if not p1_in_canvas and not p2_in_canvas:
                    continue

                ray0 = verts[edge[0]] - self.camera.get_transform().location
                ray1 = verts[edge[1]] - self.camera.get_transform().location
                cam_forward_vec = self.camera.get_transform().get_forward_vector()

                # One of the vertex is behind the camera
                if not (cam_forward_vec.dot(ray0) > 0):
                    p1 = get_image_point(verts[edge[0]], self.K_b, world_2_camera)
                if not (cam_forward_vec.dot(ray1) > 0):
                    p2 = get_image_point(verts[edge[1]], self.K_b, world_2_camera)

                cv2.line(
                    img,
                    (int(p1[0]),int(p1[1])),
                    (int(p2[0]),int(p2[1])),
                    (255,0,0, 255),
                    1
                )
        else:
            p1 = get_image_point(bb.location, self.K, world_2_camera)
            verts = [v for v in bb.get_world_vertices(npc_transform)]
            x_max = -10000
            x_min = 10000
            y_max = -10000
            y_min = 10000

            # Find the 2D bounding box by projecting all 8 vertices
            for vert in verts:
                p = get_image_point(vert, self.K, world_2_camera)
                # Find the rightmost vertex
                if p[0] > x_max:
                    x_max = p[0]
                # Find the leftmost vertex
                if p[0] < x_min:
                    x_min = p[0]
                # Find the highest vertex
                if p[1] > y_max:
                    y_max = p[1]
                # Find the lowest vertex
                if p[1] < y_min:
                    y_min = p[1]
            
            # Clip to image boundaries
            x_min = np.clip(x_min, 0, self.image_w)
            y_min = np.clip(y_min, 0, self.image_h)
            x_max = np.clip(x_max, 0, self.image_w)
            y_max = np.clip(y_max, 0, self.image_h)
            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.int32)

            invalid = False
            if min_dist:
                min_distance = min_distance_in_bbox(x, y, z, bbox, self.K)
                if min_distance >= 1000 or (x_max - x_min) * (y_max - y_min) \
                >= self.image_w * self.image_h * 0.8:
                    # Skip invalid boxes
                    invalid = True
                else:
                    # Draw the annotation
                    annotation = f"{min_distance:.1f}m"
                    (text_width, text_height), _ = cv2.getTextSize(
                        annotation, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        img, (int(x_min), int(y_min) - text_height - 4),
                        (int(x_min) + text_width, int(y_min)), (0, 255, 0), -1
                    )
                    cv2.putText(
                        img, annotation, (int(x_min), int(y_min) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1
                    )
            if not invalid:
                # Draw the bounding box
                line_width = 2
                cv2.line(
                    img, (int(x_min),int(y_min)), (int(x_max),int(y_min)),
                    (0,0,255, 255), line_width)
                cv2.line(
                    img, (int(x_min),int(y_max)), (int(x_max),int(y_max)),
                    (0,0,255, 255), line_width)
                cv2.line(
                    img, (int(x_min),int(y_min)), (int(x_min),int(y_max)),
                    (0,0,255, 255), line_width)
                cv2.line(
                    img, (int(x_max),int(y_min)), (int(x_max),int(y_max)),
                    (0,0,255, 255), line_width)
    return img


def init_module_variables(agent):
    pass

def add_module_to_agent(agent):
    '''Adds all module functionalities to an object.
    Must be called in __init__.
    '''
    init_module_variables(agent)
    # Functions
    agent.get_boxes = get_boxes.__get__(agent)
    agent.draw_lidar_boxes = draw_lidar_boxes.__get__(agent)
    agent.draw_boxes = draw_boxes.__get__(agent)
