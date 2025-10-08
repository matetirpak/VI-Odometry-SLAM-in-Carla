import numpy as np

from src.helpers.feature_matching import *

def step_vo_trajectory(agent, rgb_frame):
    kp_new, des_new = get_image_features(rgb_frame)
    agent.kp.append(kp_new)
    agent.des.append(des_new)
    if len(agent.kp) > 1:
        # Estimate rotation and translation between last two frames
        matches = match_two_images(
            agent.des[-2],
            agent.des[-1],
            threshold=0.8
        )
        rmat, tvec = estimate_movement(
            matches,
            agent.kp[-2],
            agent.kp[-1],
            agent.K,
            agent.depth_map
        )
        
        # RT for t-1 -> t
        local_RT = np.eye(4)
        local_RT[:3, :3] = rmat
        local_RT[:3, 3] = tvec.flatten()

        # Update global RT
        RT = agent.RT @ np.linalg.pinv(local_RT)
        agent.RTs.append(RT.copy())
        agent.RT = RT.copy()

        # Update trajectory
        T = RT[:3, 3].reshape((3,1))
        agent.vo_trajectory = np.hstack([agent.vo_trajectory, T])


def init_module_variables(agent):
    agent.vo_trajectory = np.zeros((3,1))
    agent.kp = []
    agent.des = []


def add_module_to_agent(agent):
    '''Adds all module functionalities to an object.
    Must be called in __init__.
    '''
    init_module_variables(agent)
    # Functions
    agent.step_vo_trajectory = step_vo_trajectory.__get__(agent)