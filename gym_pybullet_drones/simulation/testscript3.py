import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ObservationType, ActionType
from gym_pybullet_drones.envs.ObjectPickupAviary import ObjectPickupAviary  # Assuming ObjectPickupAviary is defined in this module

class Simulation:
    def __init__(self, num_steps=1000, gui=True):
        self.num_steps = num_steps
        self.gui = gui
        self.env = None

    def setup_environment(self):
        self.env = ObjectPickupAviary(drone_model=DroneModel.CF2X,
                                      physics=Physics.PYB,
                                      obs=ObservationType.KIN,
                                      act=ActionType.RPM,
                                      gui=self.gui)
        self.obs, _ = self.env.reset()

    def run(self):
        if self.env is None:
            raise ValueError("Environment not set up. Call setup_environment() first.")

        for step in range(self.num_steps):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            
            battery_level = self.env.battery.level
            print(f"Step: {step}, Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}, Battery: {battery_level}")

            if terminated or truncated:
                print("Episode finished")
                break

        self.env.close()

if __name__ == "__main__":
    sim = Simulation(num_steps=10000, gui=True)
    sim.setup_environment()
    sim.run()
