import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ObservationType, ActionType
from gym_pybullet_drones.envs.ObjectPickupAviary import ObjectPickupAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

def test_object_pickup_aviary():
    # Define initial positions for the drones
    INIT_XYZS = np.array([[.5, 0, 1]])
    # Initialize the ObjectPickupAviary environment
    env = ObjectPickupAviary(drone_model=DroneModel.CF2X,
                             initial_xyzs=INIT_XYZS,
                             physics=Physics.PYB,
                             pyb_freq=240,
                             ctrl_freq=48,
                             gui=False,
                             record=False,
                             obs=ObservationType.KIN,
                             act=ActionType.RPM)
    
    # Reset the environment
    obs, _ = env.reset()
    print("Environment initialized and reset.")
    print("Initial observation:", obs)

    # Take a single step
    action = np.zeros(4)  # Dummy action
    obs, reward, terminated, truncated, info = env.step(action)
    print("Step taken.")
    print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

if __name__ == "__main__":
    test_object_pickup_aviary()