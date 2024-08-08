import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.PickupBatteryAviary import PickupBatteryAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONE, 
        gui=DEFAULT_GUI, 
        record_video=DEFAULT_RECORD_VIDEO, 
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ, 
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, 
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        plot=True,
        colab=DEFAULT_COLAB
    ):
    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[0, 0, 1], [1, 0, 1]])
    RUBBER_DUCK_POSITION = np.array([-0.5, -0.5, 0.1])
    SPHERE_POSITION = np.array([1, 1, 0.5])
    
    env = PickupBatteryAviary(drone_model=drone,
                              num_drones=2,
                              initial_xyzs=INIT_XYZS,
                              physics=Physics.PYB_DW,
                              neighbourhood_radius=10,
                              pyb_freq=simulation_freq_hz,
                              ctrl_freq=control_freq_hz,
                              gui=gui,
                              record=record_video,
                              obstacles=True,
                              pickup_item_position=RUBBER_DUCK_POSITION
                              )

    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(drone_model=drone) for i in range(2)]

    #### Define waypoints ######################################
    waypoints = np.array([
        [RUBBER_DUCK_POSITION[0], RUBBER_DUCK_POSITION[1], INIT_XYZS[0, 2]],
        [SPHERE_POSITION[0], SPHERE_POSITION[1], INIT_XYZS[0, 2]]
    ])
    wp_counters = np.zeros(env.NUM_DRONES, dtype=int)

    #### Run the simulation ####################################
    action = np.zeros((2, 4))
    START = time.time()

    for i in range(0, 10):  # Run for 100 timesteps

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(2):
            target_pos = waypoints[wp_counters[j]]
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                 state=obs[j],
                                                                 target_pos=target_pos)
            
            # Check if the drone has reached the waypoint
            if np.linalg.norm(env.pos[j, :] - target_pos) < 0.2:
                wp_counters[j] = (wp_counters[j] + 1) % len(waypoints)

        #### Log detailed information ##############################
        print(f"Step {i}: Actions: {action}")
        for j in range(2):
            print(f"Drone {j} - Position: {env.pos[j]} Battery: {env.battery_levels[j]} Carrying Item: {env.drone_carrying_item[j]}")
            print(f"Reward for Drone {j}: {reward[j]}")

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Pickup and Battery Management example script using PickupBatteryAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 10)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
    