import numpy as np
import gymnasium as gym
import pybullet as p
from gymnasium import spaces
import pybullet_data
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from datetime import datetime
import os

class PickupBatteryAviary(BaseAviary):
    """Custom Aviary environment for drone pickup and battery management tasks."""

    def __init__(self,
                 drone_model=DroneModel.CF2X,
                 num_drones=2,
                 neighbourhood_radius=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 pyb_freq=240,
                 ctrl_freq=240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 output_folder='results',
                 max_battery_level: float = 100.0,
                 pickup_item_weight=1.0,
                 pickup_item_position=None,
                 battery_depletion_rate: float = 0.01):
        """Initialization of the pickup and battery management environment.

        Parameters are similar to BaseAviary, with additional parameters for
        battery level and item weight/position.
        """
        self.NUM_DRONES = num_drones
        self.object_ids = []
        self.object_positions = []
        self.MAX_BATTERY_LEVEL = max_battery_level
        self.battery_levels = np.full((self.NUM_DRONES,), self.MAX_BATTERY_LEVEL)
        
        super().__init__(drone_model,
                         num_drones,
                         neighbourhood_radius,
                         initial_xyzs,
                         initial_rpys,
                         physics,
                         pyb_freq,
                         ctrl_freq,
                         gui,
                         record,
                         obstacles,
                         user_debug_gui,
                         vision_attributes,
                         output_folder)
        
        # self.MAX_BATTERY_LEVEL = max_battery_level
        self.PICKUP_ITEM_WEIGHT = pickup_item_weight
        self.PICKUP_ITEM_POSITION = pickup_item_position if pickup_item_position is not None else np.array([0, 0, 0])
        
        # self.battery_levels = np.full((self.NUM_DRONES,), self.MAX_BATTERY_LEVEL)
        self.drone_carrying_item = np.zeros(self.NUM_DRONES, dtype=bool)
        
        
        self.destination = np.array([0, 2, .5])  # Example destination position
        self.battery_depletion_rate = battery_depletion_rate 
        self.item_picked = [] 
        self.constraints = {}
        self.object_positions = []
        self.dropped_items = set()
        self.drone_item_map = {}
        
        
    def _actionSpace(self):
        """Defines the action space."""
      
    
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for _ in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        """Defines the observation space."""

                #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3          Battery 
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,        0.,           0.,           0.,            0.,          0.] for _ in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_BATTERY_LEVEL] for _ in range(self.NUM_DRONES)])
        
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)


    def _computeObs(self):
        """Returns the current observation of the environment."""
        
        obs = []
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            battery = np.array([self.battery_levels[i]])
            obs.append(np.concatenate([state, battery]))
        obs = np.array(obs)
        # print(f"Computed observation shape: {obs.shape}")
        return obs

    def _preprocessAction(self, action):
        """Pre-processes the action into motor RPMs."""
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    def _computeReward(self):
        """Computes the reward for the current step."""
        
        reward = 0
        for i in range(self.NUM_DRONES):
            
            battery_penalty = (self.MAX_BATTERY_LEVEL - self.battery_levels[i])
            reward -= battery_penalty**3
            
            target_pos = self.get_target_position(i)
            distance_to_target = np.linalg.norm(np.array(self.pos[i, :]) - target_pos)
            reward += max(0, 100 - (distance_to_target**2))
        
    
            if self.drone_carrying_item[i]:
                reward += 50 
            
            collisions = self._check_collisions()
            if np.any(collisions):
                reward -= 20         
        
        
        return reward
    

    def _computeTerminated(self):
        """Computes whether the episode is terminated."""
        terminated = np.zeros(self.NUM_DRONES, dtype=bool)
        for i in range(self.NUM_DRONES):
            if self.battery_levels[i] <= 0 or (len(self.dropped_items) == len(self.object_ids)):
                terminated[i] = True
            # if np.linalg.norm(self.pos[i, :] - self.destination) < 0.5:
            #     terminated[i] = True
        return terminated

    def _computeTruncated(self):
        """Computes whether the episode is truncated."""
        truncated = np.zeros(self.NUM_DRONES, dtype=bool)
        return False

    def _computeInfo(self):
        """Computes the additional info."""
        
        return {}

    def step(self, action):
    #     """Advances the environment by one simulation step."""
     
        
        obs, reward, terminated, truncated, info = super().step(action)
    
        
        
        for i in range(self.NUM_DRONES):
            # self.battery_levels[i] -= self.battery_depletion_rate  
            self.compute_Battery(i)
            for j in range(len(self.object_ids)):
                current_object = self.object_ids[j]
                object_position, _ = p.getBasePositionAndOrientation(j, physicsClientId=self.CLIENT)
                distance_to_object = np.linalg.norm(np.array(self.pos[i, :]) - np.array(object_position))
    
                print(f"object {self.object_ids[j]} position: {object_position}")
                print(f"drone position: {np.array(self.pos[i, :])}")
                
                if (not self.drone_carrying_item[i]) and (distance_to_object < 0.1) and not(current_object in self.dropped_items ):
                    self._attach_item(i, j) 
                    
                    
                distance_to_target_destination = np.linalg.norm(np.array(self.pos[i, :]) - self.get_target_position(i))
                if self.drone_carrying_item[i] and distance_to_target_destination < 0.1 and (current_object in self.item_picked):
                    self._drop_item(i, j)
                # print(f"distanceto target: {distance_to_target_destination}")
                # print(f"distance to object: {distance_to_object}, item: {current_object}, drone {i}")
        
       
        if self.drone_carrying_item[0]: 
            item_id = self.drone_item_map[0]
            info = p.getBodyInfo(item_id)[1].decode("utf-8")  
            # print(f"the ducks information: {info }")
            # print(f"ducks info ")
            print(f"distance to object: {distance_to_object}, item: {current_object}, drone {i}")
            
            
        print(f"numberof items {len(self.object_ids)}")
        print(f"number of items picked up: {len(self.item_picked)}")
        print(f"number of dropped items: {len(self.dropped_items)}")
        # print(self.object_ids, self.item_picked)
   
        
        reward = self._computeReward()
        print(f"reward :   {reward}")
        obs = self._computeObs()
        terminated = self._computeTerminated().tolist()
        # truncated = self._computeTruncated.tolist()
        info = self._computeInfo()
        
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment."""
        # obs, info = super().reset(seed, options)
        obs, info = super().reset(seed, options)
        # obs = self._computeObs()
        self.battery_levels = np.full((self.NUM_DRONES,), self.MAX_BATTERY_LEVEL)
        self.drone_carrying_item = np.zeros(self.NUM_DRONES, dtype=bool)
        self._destroy_constraints()
        self.item_picked = []
        self.object_ids = []
        self.object_positions = []
        self._addObstacles() 
        
        
        # print(f"Reset observation shape: {obs.shape}")
        return obs, info
    
  

        
    def _destroy_constraints(self):
        """Destroy constraints if they exist."""
        for constraint_id in list(self.constraints.values()):
            try:
                if p.getConstraintInfo(constraint_id):
                    p.removeConstraint(constraint_id)
                    print(f"Constraint {constraint_id} successfully removed.")
                else:
                    # print(f"Constraint {constraint_id} does not exist.")
                    pass
            except p.error as e:
                # print(f"Warning: Failed to remove constraint {constraint_id}. Error: {e}")
                pass
        
        self.constraints.clear()

        
    
    def _reset_objects(self):
        # Reset each object's position
        for obj_id in self.object_ids:
            p.resetBasePositionAndOrientation(obj_id, [0, 0, 0], [0, 0, 0, 1])
    
    
    def _addObstacles(self):
        """Add obstacles to the environment."""
        if self.object_ids:
            for obj_id in self.object_ids:
                p.removeBody(obj_id, physicsClientId=self.CLIENT)
            self.object_ids = []
            self.object_positions = []

            

       
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Use pybullet_data's default URDFs
       
       
        duck_positions = [
        [0, 0, 0],
        [0.1, 0.1, 0],
        [0.2, 0.2, 0]  # Changed to a different position to avoid overlap
        ]

        # Load ducks
        for pos_d in duck_positions:
            duck = p.loadURDF("duck_vhacd.urdf",
                            pos_d,
                            p.getQuaternionFromEuler([0, 0, 0]),
                            physicsClientId=self.CLIENT)
            self.object_ids.append(duck)
            self.object_positions.append(pos_d)  
            
                  
        teddy_bear_positions = [
        [0, 1, 0],
        [0.1, 1.1, 0],
        [-0.1, 0.9, 0],
        [0.2, 1.2, 0],
        [-0.2, 0.8, 0] 
        ]

        for pos in teddy_bear_positions:
            teddy_bear = p.loadURDF("teddy_vhacd.urdf",
                                    pos,
                                    p.getQuaternionFromEuler([0, 0, 0]),
                                    physicsClientId=self.CLIENT)
            self.object_ids.append(teddy_bear)
            self.object_positions.append(pos)
    

        # p.loadURDF("sphere2.urdf",
        #            [1, 1, 0.5],
        #            p.getQuaternionFromEuler([0, 0, 0]),
        #            physicsClientId=self.CLIENT)
    
    def _load_object(self, position):
        size = np.random.uniform(low=0.02, high=self.object_size_threshold)
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=size, rgbaColor=[0, 1, 0, 1], physicsClientId=self.CLIENT
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE, radius=size, physicsClientId=self.CLIENT
        )
        obj_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            physicsClientId=self.CLIENT
        )
        return obj_id
    
    
    def _attach_item(self, drone_index, object_index):
        """Attach the item to the specified drone."""

        
        
        item_id = self.object_ids[object_index]  
        drone_id = self.DRONE_IDS[drone_index]
        
        if item_id in self.item_picked:
            return
        
        # Mass and size of item
        size, mass = self.item_properties(item_id)

        
        max_mass = 10  
        max_size = 10  

        
        if mass > max_mass or any(dim > max_size for dim in size):
            # print(f"Item {item_id} is too heavy or too big to be picked up by drone {drone_index}.")
            return
        
        # Get the position of the drone
        # drone_pos, _ = p.getBasePositionAndOrientation(drone_id, physicsClientId=self.CLIENT)
        
        # Move the item to the drone's position
        # p.resetBasePositionAndOrientation(item_id, drone_pos, [0, 0, 0, 1], physicsClientId=self.CLIENT)
        
        
        # Create a constraint to attach the item to the drone
        constraint_id = p.createConstraint(parentBodyUniqueId=drone_id,
                        parentLinkIndex=-1,
                        childBodyUniqueId=item_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        # childFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0],
                        physicsClientId=self.CLIENT)
        
        
        
        self.constraints[drone_index] = constraint_id
        self.drone_carrying_item[drone_index] = True
        self.item_picked.append(item_id)
        self.drone_item_map[drone_index] = item_id
        print(f"Drone {drone_index} picked up item {item_id}")
        

    
    def _drop_item(self, drone_index, object_index):
        """Drop the item from the specified drone."""
        item_id = self.object_ids[object_index]  
        drone_id = self.DRONE_IDS[drone_index]

        # Check if a constraint exists between the drone and the item
        if self.drone_carrying_item[drone_index]:
           
            constraint_id = self.constraints.get(drone_index)
            
            if constraint_id is not None:
                try:
                    p.removeConstraint(constraint_id, physicsClientId=self.CLIENT)
                    print(f"Constraint {constraint_id} removed for drone {drone_index} and item {item_id}")
                except p.error as e:
                    print(f"Warning: Failed to remove constraint {constraint_id}. Error: {e}")
                    return
                
                p.resetBaseVelocity(item_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0], physicsClientId=self.CLIENT)
                
                # Set the drone_carrying_item flag to False
                self.drone_carrying_item[drone_index] = False
                self.item_picked.remove(item_id)
                self.dropped_items.add(item_id)
                del self.drone_item_map[drone_index]
                
                print(f"Drone {drone_index} dropped item {item_id}")
            else:
                print(f"No constraint found for drone {drone_index} and item {item_id}")

    
    
    def get_target_position(self, drone_index):
        
        
        if self.drone_carrying_item[drone_index]:
            item_id = self.drone_item_map[drone_index]
            if "cube" in p.getBodyInfo(item_id)[1].decode("utf-8"):
                return np.array([0, 1, 0])  
            else:
                return np.array([0, 0, 0]) 
        else:
            for j in range(len(self.object_ids)):
                # object_pos, _ = p.getBasePositionAndOrientation(j, physicsClientId=self.CLIENT)
                current_object = self.object_ids[j]
                if current_object not in self.item_picked and current_object not in self.dropped_items:
                    object_pos, _ = p.getBasePositionAndOrientation(j, physicsClientId=self.CLIENT)
                    return np.array(object_pos)
                
            return np.array([0, 0, 0])

        
    
    def item_properties(self, item_id):
        
        # Retrieve the mass of the item
        mass = p.getDynamicsInfo(item_id, -1)[0]

        # size of the objects
        aabb_min, aabb_max = p.getAABB(item_id)
        size = [aabb_max[i] - aabb_min[i] for i in range(3)]
        

        # print(f"mass = {mass} kg, size = {size}")
        
        return size, mass
    
    
    def compute_Battery(self, drone_index):
        """Adjusts the battery depletion rate based on the mass of the item being carried."""
        base_depletion_rate = self.battery_depletion_rate
        additional_depletion_rate = 0
        
        if self.drone_carrying_item[drone_index]:
            item_id = self.drone_item_map[drone_index]
            _, mass = self.item_properties(item_id)
            additional_depletion_rate = mass * 0.1  # Adjust the multiplier as needed
            
        total_depletion_rate = base_depletion_rate + additional_depletion_rate
        self.battery_levels[drone_index] -= total_depletion_rate
    
    
    
    def _check_collisions(self):
        """Check for collisions in the environment."""
        num_bodies = p.getNumBodies(physicsClientId=self.CLIENT)  
        collisions = np.zeros(3)  # Placeholder for collision information

        for i in range(num_bodies):
            
            contact_points = p.getContactPoints(bodyA=i, bodyB=0, physicsClientId=self.CLIENT)  

            # If there is a collision
            if contact_points:
                
                collisions = np.ones(3)  # Update collision information based on your needs
                break  # Exit loop on first collision found

        return collisions
        
    
        
        
        
            
        