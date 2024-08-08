import numpy as np
import pybullet as p
import gymnasium as gym
from gym import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.utils import Battery

class ObjectPickupAviary(BaseRLAviary):
    MASS = 0.5
    max_distance=0.32
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_ids = []
        self.object_size_threshold = 0.1  # Maximum object size
        self.target_object_size = 0.05  # Ideal object size
        self.object_positions = []  # Initialize here
        self.battery = Battery(initial_level=1.0, max_level=1.0, min_level=0.0, discharge_rate=0.01)
        self.previous_distance_to_object = 0.0
        self.EPISODE_LEN_SEC = 5  # Define episode length in seconds
        self.target_pos = self._generate_random_target_position()
        self.target_vel = self._generate_random_target_velocity()

        # Define observation space with finite boundaries for ALL elements
        self.observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(4,), dtype=np.float32)

        # Continuous action space for RPMs
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _generate_random_target_position(self):
        """Generate a random target position within the simulation space bounds."""
        x_range = (-5, 5)  # Define the range for x coordinate
        y_range = (-5, 5)  # Define the range for y coordinate
        z_range = (0, 1)   # Define the range for z coordinate, usually above the ground level

        target_pos = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            np.random.uniform(*z_range)
        ])
        return target_pos

    def _generate_random_target_velocity(self):
        """Generate a random target velocity."""
        vel_range = (-1, 1)  # Define the range for each component of the velocity

        target_vel = np.array([
            np.random.uniform(*vel_range),
            np.random.uniform(*vel_range),
            np.random.uniform(*vel_range)
        ])
        return target_vel

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set the random seed for reproducibility

        super().reset()  # Call the reset method from the BaseRLAviary

        self._add_objects()
        self.battery.level = 1.0
        self.previous_distance_to_object = np.inf

    # Update self.object_positions here after adding objects
        self.object_positions = [
            p.getBasePositionAndOrientation(obj_id, physicsClientId=self.CLIENT)[0]
            for obj_id in self.object_ids
        ]

        obs = self._compute_obs()  
        reset_info = {}  
        
            # Print debug information
        print(f"Initial drone position: {self._getDroneState(0)[:3]}")
        print(f"Target position: {self.target_pos}")
        print(f"Object positions: {self.object_positions}")
        print(f"Initial battery level: {self.battery.level}")

        return obs, reset_info



    def _add_objects(self):
            
        if self.object_ids:
            for obj_id in self.object_ids:
                p.removeBody(obj_id, physicsClientId=self.CLIENT)
            self.object_ids = []
            self.object_positions = []
            
        # self.object_ids.append(self._load_object(position=[0, 0, 0.1], size=0.05, color=[1, 0, 0, 1], shape=p.GEOM_BOX, half_extents=[0.1, 0.01, 0.01]))  # Pencil
        # self.object_ids.append(self._load_object(position=[1, 0, 0.1], size=0.05, color=[0, 1, 0, 1], shape=p.GEOM_BOX, half_extents=[0.02, 0.02, 0.02]))  # Sharpener
        # self.object_ids.append(self._load_object(position=[0, 1, 0.1], size=0.1, color=[1, 1, 0, 1], shape=p.GEOM_SPHERE))  # Rubber Duck
        # self.object_ids.append(self._load_object(position=[-1, -1, 0.1], size=0.2, color=[0, 0, 1, 1], shape=p.GEOM_BOX, half_extents=[0.2, 0.2, 0.1]))  # Larger Box
        # self.object_ids.append(self._load_object(position=[0, -1, 0.4], size=0.08, color=[1.5, 0, 0, 1], shape=p.GEOM_BOX, half_extents=[0.1, 0.01, 0.01])) 
        # self.object_ids.append(self._load_object(position=[-1, 0, 0.7], size=0.09, color=[2, 0, 0, 1], shape=p.GEOM_BOX, half_extents=[0.1, 0.01, 0.01])) 
             
        # Add new objects
        for _ in range(5):
            while True:
                position = self._generate_random_position()
                if all(np.linalg.norm(position - existing_pos) > self.object_size_threshold * 2 
                       for existing_pos in self.object_positions):
                    break
            self.object_positions.append(position)
            self.object_ids.append(self._load_object(position))
        



    def _generate_random_position(self):
        return np.random.uniform(low=[-1, -1, 0.1], high=[1, 1, 0.1])

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
    
    def _compute_obs(self):
        nth_drone = 0  # Assuming nth_drone is 0 for now
        drone_state = self._getDroneState(nth_drone)

        if len(drone_state) < 13:
        # Handle case where drone_state does not have enough elements
            
            return np.zeros_like(self.observation_space.low, dtype=np.float32)

        target_pos = self.target_pos
        target_vel = self.target_vel

        target_rel_pos = target_pos - drone_state[0:3]
        target_rel_vel = target_vel - drone_state[10:13]

    # Example: Assuming collisions is a placeholder array
        collisions = self._check_collisions()

        obs = np.concatenate([drone_state, target_pos, target_vel, target_rel_pos, target_rel_vel, collisions])
        return obs


    def _check_collisions(self):
        """Check for collisions in the environment."""
        num_bodies = p.getNumBodies()  # Get the number of bodies in the simulation
        collisions = np.zeros(3)  # Placeholder for collision information

        for i in range(num_bodies):
            # Check for collisions involving drones or specific objects of interest
            # Example: Check collisions with the ground plane
            contact_points = p.getContactPoints(bodyA=i, bodyB=0)  # Assuming ground plane ID is 0
            
            if contact_points:
                # There is a collision
                collisions = np.ones(3)  # Update collision information based on your needs
                break  # Exit loop on first collision found
        
        return collisions

    def step(self, action):
        obs = self._compute_obs()
        reward = self._compute_reward(action)
        done = self._compute_done()
        info = self._computeInfo()
        terminated = done
        truncated = False
        # Ensure 'episode' key exists in info dictionary
        if not isinstance(info, dict):
            info = {}
        info["episode"] = {}  # Add any relevant episode info here if needed

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action):
        reward = 0

        # Ensure there are object positions before calculating the closest object distance
        if self.object_positions:
            closest_object_distance = np.min([
                np.linalg.norm(np.array(obj_pos) - self._getDroneStateVector(0)[:3])
                for obj_pos in self.object_positions
            ])
            reward += 1 / (1 + closest_object_distance)  # Closer -> Higher reward
        else:
            closest_object_distance = np.inf

        # Battery consumption penalty (scale based on continuous actions)
        rpm_scaling_factor = 0.25
        max_rpm = 9.81 * self.MASS / 4  # Maximum RPM for hovering
        current_rpm = action * max_rpm  # Scale continuous actions to RPM

        # Adjust battery drain rate based on average RPM
        avg_rpm = np.mean(current_rpm)
        battery_discharge_rate = 0.01 + 0.02 * (avg_rpm / max_rpm) 
        reward -= battery_discharge_rate
        self.battery.level -= battery_discharge_rate

        if self.battery.level <= 0:
            reward -= 10  # Penalty for battery depletion
            self.battery.level = 0  # Prevent negative battery level

        return reward


    def _compute_done(self):
        
        # distance_to_target = np.linalg.norm(self.target_pos - self._getDroneState(0)[:3])
        
        # if distance_to_target < 0.1 and not self.carrying_object:
        #     self.target_reached = True
        #     print("Terminating because the drone reached the target position.")
        #     return True

        # if distance_to_target > self.max_distance or self.battery.level <= 0:
        #     print(f"Terminating because distance to target is too far: {distance_to_target > self.max_distance} or battery depleted: {self.battery.level <= 0}")
        #     return True

        return False


    def _getTargetState(self):
        target_state = np.array(self.target_pos, dtype=np.float32)
        return target_state


    def _getDroneState(self, drone_index=0):
        # Implement logic to retrieve and return drone state vector
        # Example implementation:
        position, orientation = p.getBasePositionAndOrientation(self.DRONE_IDS[drone_index], physicsClientId=self.CLIENT)
        velocity, angular_velocity = p.getBaseVelocity(self.DRONE_IDS[drone_index], physicsClientId=self.CLIENT)
        euler_angles = p.getEulerFromQuaternion(orientation)

        battery_level = self.battery.level

    # Check if the drone is carrying an object
        carrying_object = 1 if len(self.object_ids) > 0 else 0

    # Calculate distance to target
        distance_to_target = np.linalg.norm(self.target_pos - position)

        

        drone_state = np.array([
            position[0], position[1], position[2],  # Position components
            velocity[0], velocity[1], velocity[2],  # Velocity components
            euler_angles[0], euler_angles[1], euler_angles[2],  # Orientation (Euler angles)
            battery_level,  # Battery level
            carrying_object,  # Binary flag indicating if the drone is carrying an object (1) or not (0)
            distance_to_target,  # Battery level
            
            
        ], dtype=np.float32)

        return drone_state

    def _computeInfo(self):
        """Compute and return additional information for the environment."""
        info={}

        
        drone_states = []
        for drone_index in range(len(self.DRONE_IDS)):
            drone_state = self._getDroneState(drone_index)
            drone_states.append(drone_state)

        info['drone_states'] = drone_states  # Store drone states in info dictionary

        return info



    def make_env(env_id, rank, seed=0):
        def _init():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
        return _init

    def _checkObjectPickup(self, action):
        drone_position = self._getDroneStateVector(0)[:3]
        for obj_id, obj_pos in zip(self.object_ids, self.object_positions):
            distance_to_object = np.linalg.norm(np.array(obj_pos) - drone_position)
            if distance_to_object < 0.1:
                size = p.getVisualShapeData(obj_id, physicsClientId=self.CLIENT)[0][3][0]
                if size <= self.target_object_size:
                    p.removeBody(obj_id, physicsClientId=self.CLIENT)
                    self.object_ids.remove(obj_id)
                    self.object_positions.remove(obj_pos)
                    self._addObjects()  # Add a new object

                # Convert continuous actions to RPMs and apply force
                    rpm_scaling_factor = 0.25
                    max_rpm = 9.81 * self.MASS / 4  # Maximum RPM for hovering
                    current_rpm = action * max_rpm  # Scale continuous actions to RPM
                    p.applyExternalForce(self.DRONE_IDS[0], -1, forceObj=[0, 0, sum(current_rpm)], 
                                    posObj=drone_position, flags=p.WORLD_FRAME, physicsClientId=self.CLIENT)



                    return True  # Successfully picked up
                else:
                    return False  # Object too large
        return False  # No object close enough
    
    def _preprocessAction(self, action):
        """Pre-processes the action input."""
        self.battery.level -= self.battery_consumption_rate
        return super()._preprocessAction(action)

