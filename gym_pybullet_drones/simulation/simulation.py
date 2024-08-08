
import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ObservationType, ActionType
from gym_pybullet_drones.envs.ObjectPickupAviary import ObjectPickupAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12

def run(
        drone=DEFAULT_DRONE, 
        gui=DEFAULT_GUI, 
        record_video=DEFAULT_RECORD_VIDEO, 
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ, 
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, 
        duration_sec=DEFAULT_DURATION_SEC
    ):
    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[.5, 0, 1]])
    env = ObjectPickupAviary(drone_model=drone,
                             initial_xyzs=INIT_XYZS,
                             physics=Physics.PYB,
                             pyb_freq=simulation_freq_hz,
                             ctrl_freq=control_freq_hz,
                             gui=gui,
                             record=record_video,
                             obs=ObservationType.KIN,
                             act=ActionType.RPM
                             )

    #### Initialize the controller #############################
    ctrl = DSLPIDControl(drone_model=drone)

    #### Run the simulation ####################################
    action = np.zeros((1,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step: {i}, Reward: {reward}, Done: {terminated}, Info: {info}")

        #### Compute control for the current observation #############
        target_pos = np.array([0, 0, 1])  # Dummy target position
        action[0, :], _, _ = ctrl.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                          state=obs,
                                                          target_pos=target_pos
                                                          )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

        #### Check if episode is done ##############################
        if terminated:
            print("Episode finished.")
            break

    #### Close the environment #################################
    env.close()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Object pickup example script using ObjectPickupAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 12)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))