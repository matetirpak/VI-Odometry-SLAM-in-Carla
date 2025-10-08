import numpy as np
import matplotlib.pyplot as plt
import cv2


def step_gt_trajectory(agent):
    """
    Advances the ground truth trajectory by one step
    using the ego vehicle's current position.
    """
    # Extract location
    transform = agent.ego_vehicle.get_transform()
    location = transform.location
    world_coords = np.array([
        [location.x],
        [location.y],
        [location.z+1.7],
        [1.]
    ])
    # Convert to camera coordinates
    camera_coords = agent.world2camera @ world_coords
    camera_coords = np.array([
        [camera_coords[1,0]],
        [-camera_coords[2,0]],
        [camera_coords[0,0]]
    ])

    # Update trajectory
    agent.gt_trajectory = np.hstack([
        agent.gt_trajectory, 
        camera_coords - agent.initial_offset
    ])

def init_odometry_plot(agent):
    """
    Initializes the matplotlib figure and axes for trajectory plotting.
    """
    agent.fig = plt.figure(figsize=(15,4))
    agent.ax = agent.fig.add_subplot(1,4,1, projection='3d')
    agent.ax2 = agent.fig.add_subplot(1,4,2)
    agent.ax2.set_aspect('equal', adjustable='box')
    agent.ax3 = agent.fig.add_subplot(1,4,3)
    agent.ax3.set_aspect('equal', adjustable='box')
    agent.ax4 = agent.fig.add_subplot(1,4,4)
    agent.ax4.set_aspect('equal', adjustable='box')

def draw_trajectory(agent, estimated, actual, figname="trajectory"):
    """
    Updates trajectory plot in-place.
    """
    plt.cla()
    agent.ax.clear()
    agent.ax.plot3D(estimated[0, :], estimated[1, :], estimated[2, :], 'r', label='estimated')
    agent.ax.scatter3D(estimated[0, :], estimated[1, :], estimated[2, :], c='r')
    agent.ax.plot3D(actual[0, :], actual[1, :], actual[2, :], 'b', label='actual')
    agent.ax.scatter3D(actual[0, :], actual[1, :], actual[2, :], c='b')
    agent.ax.set_title('3D')
    agent.ax.set_xlabel('X')
    agent.ax.set_ylabel('Y')
    agent.ax.set_zlabel('Z')
    agent.ax.legend()

    agent.ax2.clear()
    agent.ax2.plot(estimated[0, :], estimated[1, :], 'r', label='estimated')
    agent.ax2.scatter(estimated[0, :], estimated[1, :], c='r')
    agent.ax2.plot(actual[0, :], actual[1, :], 'b', label='actual')
    agent.ax2.scatter(actual[0, :], actual[1, :], c='b')
    agent.ax2.set_xlabel('X')
    agent.ax2.set_ylabel('Y')
    agent.ax2.set_title('x-y')

    agent.ax3.clear()
    agent.ax3.plot(estimated[0, :], estimated[2, :], 'r', label='estimated')
    agent.ax3.scatter(estimated[0, :], estimated[2, :], c='r')
    agent.ax3.plot(actual[0, :], actual[2, :], 'b', label='actual')
    agent.ax3.scatter(actual[0, :], actual[2, :], c='b')
    agent.ax3.set_xlabel('X')
    agent.ax3.set_ylabel('Z')
    agent.ax3.set_title('x-z')

    agent.ax4.clear()
    agent.ax4.plot(estimated[1, :], estimated[2, :], 'r', label='estimated')
    agent.ax4.scatter(estimated[1, :], estimated[2, :], c='r')
    agent.ax4.plot(actual[1, :], actual[2, :], 'b', label='actual')
    agent.ax4.scatter(actual[1, :], actual[2, :], c='b')
    agent.ax4.set_title('y-z')
    agent.ax4.set_xlabel('Y')
    agent.ax4.set_ylabel('Z')

    agent.fig.canvas.draw()
    fig_img = np.fromstring(agent.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    fig_img = fig_img.reshape(agent.fig.canvas.get_width_height()[::-1] + (3,))
    _show_frame(cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR), name=figname)

def _show_frame(img, name='frame'):
    cv2.imshow(name, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return 0

def init_module_variables(agent):
    agent.gt_trajectory = np.zeros((3,1))
    agent.RT = np.eye(4)
    agent.RTs = [agent.RT]
    agent.left_to_right_hand_coord_system = np.array([
        [ 0,  1,  0,  0],
        [ 0,  0, -1,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1],
    ])
    agent.world2camera = np.array(agent.camera.get_transform().get_inverse_matrix())
    location = agent.camera.get_transform().location
    world_start = np.array([
        [location.x],
        [location.y],
        [location.z],
        [1.]
    ])
    agent.initial_offset = (agent.world2camera @ world_start)

    agent.initial_offset = np.array([
        [agent.initial_offset[1,0]],
        [-agent.initial_offset[2,0]],
        [agent.initial_offset[0,0]]
    ])


def add_module_to_agent(agent):
    '''Adds all module functionalities to an object.
    Must be called in __init__.
    '''
    init_module_variables(agent)
    # Functions
    agent.step_gt_trajectory = step_gt_trajectory.__get__(agent)
    agent.init_odometry_plot = init_odometry_plot.__get__(agent)
    agent.draw_trajectory = draw_trajectory.__get__(agent)


