from websockets.sync.client import connect
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import json


class ShiftWrapper(gym.Wrapper):
    """Allow to use Discrete() action spaces with start!=0"""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = gym.spaces.Discrete(env.action_space.n, start=0)

    def step(self, action: int):
        return self.env.step(action + self.env.action_space.start)


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, startingBlocks: int = 3, dim: int = 10, distToBorder: int = 3, 
                 ws_url: str = "ws://localhost:3030"):
        super().__init__()
        self.startingBlocks = startingBlocks
        self.dim = dim
        self.distToBorder = distToBorder
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=3,
                                            shape=(dim, dim), dtype=np.int8)
        self.ws_url = ws_url
        self.sentinelValue = -111
        
        self._connect()
    
    def _connect(self):
        """Establish websocket connection with error handling"""
        try:
            self.websocket = connect(self.ws_url)
            print(f"Connected to {self.ws_url}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to websocket: {e}")
    
    def receiveReward(self, prefix="reward"):
        """Receive and parse reward from websocket"""
        try:
            reward_str = str(self.websocket.recv())
            reward_str = reward_str.removeprefix(prefix)
            print(f"Received reward: {reward_str}")
            reward = float(reward_str)
            return reward
        except ValueError as e:
            print(f"Error parsing reward: {e}")
            return 0.0
        except Exception as e:
            print(f"Error receiving reward: {e}")
            return 0.0
    
    def receiveObservation(self, prefix="observation_space"):
        """Receive and parse observation from websocket"""
        try:
            obs_str = str(self.websocket.recv())
            obs_str = obs_str.removeprefix(prefix)
            print(f"Received observation: {obs_str[:100]}...")  # Print first 100 chars
            
            # Parse observation (assuming JSON format)
            # Adjust parsing based on your actual format
            obs_data = json.loads(obs_str)
            
            # Convert to numpy array matching observation_space
            if isinstance(obs_data, list):
                observation = np.array(obs_data, dtype=np.int8)
            else:
                observation = np.array(obs_data['grid'], dtype=np.int8)
            
            # Ensure correct shape
            if observation.shape != (self.dim, self.dim):
                observation = observation.reshape((self.dim, self.dim))
            
            return observation
        except Exception as e:
            print(f"Error receiving/parsing observation: {e}")
            # Return zero observation as fallback
            return np.zeros((self.dim, self.dim), dtype=np.int8)
    
    def sendAction(self, action: int, prefix="action"):
        """Send action to websocket"""
        try:
            action_msg = f"{prefix}{int(action)}"
            self.websocket.send(action_msg)
            print(f"Sent action: {action}")
        except Exception as e:
            print(f"Error sending action: {e}")
            raise
        
    def step(self, action: int):
        """Execute one time step within the environment"""
        self.sendAction(action=action)
        
        reward = self.receiveReward()
        observation = self.receiveObservation()
        
        print(f"received this reward: {reward}")
        
        # Check termination condition
        terminated = (reward == self.sentinelValue)
        truncated = False  # Add your truncation logic here (e.g., max steps)
        info = {}  # Add any additional info here
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        # Set seed if provided
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        try:
            # Send reset command
            reset_msg = "reset"
            self.websocket.send(reset_msg)
            print("Sent reset command")
            
            # Receive initial observation
            observation = self.receiveObservation()
            info = {}
            
            return observation, info
        except Exception as e:
            print(f"Error during reset: {e}")
            # Return zero observation as fallback
            return np.zeros((self.dim, self.dim), dtype=np.int8), {}
    
    def close(self):
        """Clean up resources"""
        if self.websocket:
            try:
                self.websocket.close()
                print("Websocket connection closed")
            except Exception as e:
                print(f"Error closing websocket: {e}")
        super().close()
    
    def render(self):
        """Render the environment (optional implementation)"""
        pass
    
