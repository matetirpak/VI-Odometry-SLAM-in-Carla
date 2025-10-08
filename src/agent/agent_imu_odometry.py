import numpy as np
from scipy.spatial.transform import Rotation as R

class IMU:
    def __init__(self):
        # Estimated state variables
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.orientation = R.identity()

    def update(self, accel_local, gyro_local, dt):
        # Integrate (rad/s) gyroscope
        delta_angle = gyro_local * dt
        # Angular velocity to rotation vector
        delta_rotation = R.from_rotvec(delta_angle)
        # Update orientation
        self.orientation = self.orientation * delta_rotation
        M = self.orientation.as_matrix()

        # Rotate local acceleration to world frame
        accel_world = self.orientation.apply(accel_local)
        # Subtract gravity
        accel_world[2] -= 9.81

        # Integrate to update velocity and position
        self.position += self.velocity * dt + 0.5 * accel_world * dt * dt
        self.velocity += accel_world * dt

    def get_position(self):
        return self.position.copy()

    def get_velocity(self):
        return self.velocity.copy()

    def get_orientation(self):
        # Returns roll, pitch, yaw in radians
        return self.orientation.as_euler('xyz', degrees=False).copy()


def update_imu_state(agent, imu_data, dt):
    # Extract IMU data
    accel = Vector3D_to_numpy(imu_data.accelerometer)
    gyro = Vector3D_to_numpy(imu_data.gyroscope)

    # Update
    agent.imu_calculator.update(accel, gyro, dt)

def step_io_trajectory(agent, imu_data, dt):
    

    # Extract position and orientation
    R = agent.imu_calculator.orientation.as_matrix()
    T_world = agent.imu_calculator.get_position()

    # Convert position to the camera coordinate system
    T = np.array([
        T_world[1],
        T_world[2]*-1,
        T_world[0],
    ]).reshape((3,1))

    agent.io_trajectory = np.hstack([agent.io_trajectory, T])

    # Update the current RT matrix
    RT = np.eye(4)
    RT[:3, 3] = T.flatten()
    RT[:3, :3] = R
    agent.RTs.append(RT.copy())
    agent.RT = RT.copy()


def Vector3D_to_numpy(vec):
    return np.array([vec.x, vec.y, vec.z])

def init_module_variables(agent):
    agent.io_trajectory = np.zeros((3,1))
    agent.imu_calculator = IMU()

def add_module_to_agent(agent):
    '''Adds all module functionalities to an object.
    Must be called in __init__.
    '''
    init_module_variables(agent)
    # Functions
    agent.update_imu_state = update_imu_state.__get__(agent)
    agent.step_io_trajectory = step_io_trajectory.__get__(agent)