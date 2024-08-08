import numpy as np
import gym
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.envs.PickupDeliveryAviary import PickupDeliveryAviary

def run_simulation():
    # Create the environment
    env = PickupDeliveryAviary(drone_model=DroneModel.CF2X,
                               physics=Physics.PYB,
                               obs=ObservationType.KIN,
                               act=ActionType.RPM,
                               gui=True)

    # Reset the environment
    obs = env.reset()
    # Define the controller
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    # Define the target position (first object)
    target_pos = np.array([-1, 0, 0.7])


    # Run the simulation for a fixed number of steps or until the episode is done
    num_steps = 100000000
    for step in range(num_steps):
        drone_pos = obs[0:3]
        drone_quat = obs[3:7]

        # Compute the control action
        action, _, _ = ctrl.computeControlFromState(control_timestep=1/env.CTRL_FREQ,
                                                    state=np.hstack([drone_pos, drone_quat]),
                                                    target_pos=target_pos)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        # Print some debug info
        print(f"Step: {step}, Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        
        # Check if the episode is done
        if terminated or truncated:
            print("Episode finished")
            break
        
        if env.item_picked_up:
            target_pos = env.TARGET_DELIVERY_POS

    # Close the environment
    env.close()

if __name__ == "__main__":
    run_simulation()