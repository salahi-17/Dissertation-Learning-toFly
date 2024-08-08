import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class PickupDeliveryAviary(BaseRLAviary):
    """Single agent RL problem: pick up and deliver items with battery level tracking."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM):
        
        """Initialization of a single agent RL environment."""
        self.TARGET_PICKUP_POS = np.array([0, 0, 1])
        self.TARGET_DELIVERY_POS = np.array([1, 1, 1])
        self.EPISODE_LEN_SEC = 15
        self.BATTERY_CAPACITY = 100
        self.battery_level = self.BATTERY_CAPACITY
        self.PICKUP_REWARD = 10
        self.DELIVERY_REWARD = 50
        self.battery_consumption_rate = 1.0
        self.item_picked_up = False
        self.object_ids = []
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    def reset(self):
        """Resets the environment."""
        super().reset()
        self.battery_level = self.BATTERY_CAPACITY
        self.item_picked_up = False
        self._add_objects()
        self.TARGET_PICKUP_POS = np.array([0, 0, 0.1])  # Set the pickup target
        self.TARGET_DELIVERY_POS = np.array([1, 1, 1])  # Set the delivery target
        obs, info = self._compute_obs()
        info = {}
        return obs, info

    def _add_objects(self):
        """Adds objects to the simulation."""
        if self.object_ids:
            for obj_id in self.object_ids:
                p.removeBody(obj_id, physicsClientId=self.CLIENT)
            self.object_ids = []

        # Add specific objects with different sizes
        self.object_ids.append(self._load_object(position=[0, 0, 0.1], size=0.05, color=[1, 0, 0, 1], shape=p.GEOM_BOX, half_extents=[0.1, 0.01, 0.01]))  # Pencil
        self.object_ids.append(self._load_object(position=[1, 0, 0.1], size=0.05, color=[0, 1, 0, 1], shape=p.GEOM_BOX, half_extents=[0.02, 0.02, 0.02]))  # Sharpener
        self.object_ids.append(self._load_object(position=[0, 1, 0.1], size=0.1, color=[1, 1, 0, 1], shape=p.GEOM_SPHERE))  # Rubber Duck
        self.object_ids.append(self._load_object(position=[-1, -1, 0.1], size=0.2, color=[0, 0, 1, 1], shape=p.GEOM_BOX, half_extents=[0.2, 0.2, 0.1]))  # Larger Box
        self.object_ids.append(self._load_object(position=[0, -1, 0.4], size=0.08, color=[1.5, 0, 0, 1], shape=p.GEOM_BOX, half_extents=[0.1, 0.01, 0.01])) 
        self.object_ids.append(self._load_object(position=[-1, 0, 0.7], size=0.09, color=[2, 0, 0, 1], shape=p.GEOM_BOX, half_extents=[0.1, 0.01, 0.01])) 

    def _load_object(self, position, size, color, shape=p.GEOM_SPHERE, half_extents=None):
        """Loads an object into the simulation."""
        if shape == p.GEOM_BOX and half_extents is not None:
            visual_shape_id = p.createVisualShape(shapeType=shape, halfExtents=half_extents, rgbaColor=color, physicsClientId=self.CLIENT)
            collision_shape_id = p.createCollisionShape(shapeType=shape, halfExtents=half_extents, physicsClientId=self.CLIENT)
        else:
            visual_shape_id = p.createVisualShape(shapeType=shape, radius=size, rgbaColor=color, physicsClientId=self.CLIENT)
            collision_shape_id = p.createCollisionShape(shapeType=shape, radius=size, physicsClientId=self.CLIENT)
        
        obj_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            physicsClientId=self.CLIENT
        )
        return obj_id

    def _computeReward(self):
        """Computes the current reward value."""
        state = self._getDroneStateVector(0)
        reward = 0
        if not self.item_picked_up:
            distance_to_pickup = np.linalg.norm(self.TARGET_PICKUP_POS - state[0:3])
            if distance_to_pickup < 0.1:
                self.item_picked_up = True
                reward += self.PICKUP_REWARD
            else:
                reward += max(0, 1 - distance_to_pickup**2)
        else:
            distance_to_delivery = np.linalg.norm(self.TARGET_DELIVERY_POS - state[0:3])
            if distance_to_delivery < 0.1:
                reward += self.DELIVERY_REWARD
            else:
                reward += max(0, 1 - distance_to_delivery**2)
        
        return reward

    def _computeTerminated(self):
        """Computes the current done value."""
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_DELIVERY_POS - state[0:3]) < 0.1 and self.item_picked_up:
            return True
        if self.battery_level <= 0:
            return True
        return False

    def _computeTruncated(self):
        """Computes the current truncated value."""
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0  # Truncate when the drone is too far away
            or abs(state[7]) > .4 or abs(state[8]) > .4):  # Truncate when the drone is too tilted
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False
    
    def _compute_obs(self):
        """Computes the current observation."""
        state = self._getDroneStateVector(0)
        obs = np.hstack([state[0:3], state[3:7], [self.battery_level], [1 if self.item_picked_up else 0]])
        return obs

    def _computeInfo(self):
        """Computes the current info dict(s)."""
        return {"answer": 42}  # Calculated by the Deep Thought supercomputer in 7.5M years

    def _preprocessAction(self, action):
        """Pre-processes the action input."""
        self.battery_level -= self.battery_consumption_rate
        return super()._preprocessAction(action)
    
    def _checkObjectPickup(self, action):
        """Checks and performs object pickup."""
        drone_position = self._getDroneStateVector(0)[:3]
        for obj_id in self.object_ids:
            obj_pos = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.CLIENT)[0]
            distance_to_object = np.linalg.norm(np.array(obj_pos) - drone_position)
            if distance_to_object < 0.1:
                size = p.getVisualShapeData(obj_id, physicsClientId=self.CLIENT)[0][3][0]
                if size <= self.target_object_size:
                    p.removeBody(obj_id, physicsClientId=self.CLIENT)
                    self.object_ids.remove(obj_id)
                    return True
        return False
    
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        if not self.item_picked_up:
            if self._checkObjectPickup(action):
                self.item_picked_up = True
                self.TARGET_PICKUP_POS = self.TARGET_DELIVERY_POS  # Set new target to delivery position
                print("Picked up object, moving to delivery point")
        
        obs = self._compute_obs()
        return obs, reward, terminated, truncated, info
    
    def _getDroneStateVector(self, nth_drone):
        """Returns the state vector of the nth drone."""
        pos, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[nth_drone], physicsClientId=self.CLIENT)
        vel, ang_vel = p.getBaseVelocity(self.DRONE_IDS[nth_drone], physicsClientId=self.CLIENT)
        state = np.hstack([pos, quat, vel, ang_vel])
        return state
