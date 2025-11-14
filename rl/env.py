from websockets.sync.client import connect
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import json
import time


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

    def __init__(self, startingBlocks: int = 3, dim: int = 20, distToBorder: int = 3, 
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
        self.websocket = None
        
        self._connect()
    
    def _connect(self):
        """Establish websocket connection with error handling"""
        try:
            # Pass close_timeout parameter to avoid hanging on close
            self.websocket = connect(self.ws_url, close_timeout=5)
            print(f"Connected to {self.ws_url}")
            # Send ping to identify as Python client
            self.websocket.send("pingfrombackend")
            time.sleep(0.5)  # Give server time to register
        except Exception as e:
            raise ConnectionError(f"Failed to connect to websocket: {e}")
    
    def _recv_message(self):
        """Receive and decode message properly"""
        try:
            msg = self.websocket.recv(timeout=1)  # Pass timeout as parameter
            # Handle both bytes and string messages
            if isinstance(msg, bytes):
                return msg.decode('utf-8')
            return str(msg)
        except TimeoutError:
            print(f"Timeout waiting for message")
            raise
        except Exception as e:
            print(f"Error receiving message: {e}")
            raise
    
    def receiveReward(self, prefix="reward"):
        """Receive and parse reward from websocket"""
        reward_str = ""
        try:
            reward_str = str(self._recv_message())
            
            # Skip non-reward messages
            while not reward_str.startswith(prefix):
                print(f"Skipping non-reward message: {reward_str[:50]}")
                reward_str = str(self._recv_message())
            
            reward_str = reward_str.removeprefix(prefix)
            print(f"Received reward: {reward_str}")
            reward = float(reward_str)
            return reward
        except ValueError as e:
            print(f"Error parsing reward: {e}, string was: {reward_str}")
            return 0.0
        except Exception as e:
            print(f"Error receiving reward: {e}")
            return 0.0
    
    def receiveObservation(self, prefix="observation_space"):
        """Receive and parse observation from websocket"""
        obs_str = ""
        try:
            obs_str = str(self._recv_message())
            
            # Skip non-observation messages
            while not obs_str.startswith(prefix):
                print(f"Skipping non-observation message: {obs_str[:50]}")
                obs_str = str(self._recv_message())
            
            obs_str = obs_str.removeprefix(prefix)
            print(f"Received observation: {obs_str[:100]}...")
            
            # Parse JSON observation
            obs_data = json.loads(obs_str)
            observation = np.array(obs_data, dtype=np.int8)
            
            # Ensure correct shape
            if observation.shape != (self.dim, self.dim):
                observation = observation.reshape((self.dim, self.dim))
            
            return observation
        except Exception as e:
            print(f"Error receiving/parsing observation: {e}")
            if obs_str:
                print(f"Observation string was: {obs_str[:200]}")
            return np.zeros((self.dim, self.dim), dtype=np.int8)
    
    def sendAction(self, action: int, prefix="action"):
        """Send action to websocket"""
        try:
            action_msg = f"{prefix}{str(action)}"
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
        
        print(f"Step complete - reward: {reward}")
        
        # Check termination condition based on reward
        terminated = (reward == -10)  # Game ends on collision
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        try:
            # For first reset, just consume initial messages
            if hasattr(self, '_first_reset_done'):
                # Send reset command for subsequent resets
                reset_msg = "reset"
                self.websocket.send(reset_msg)
                print("Sent reset command")
                
                # Wait for game to reload and reconnect
                time.sleep(2.0)
                
                # Reconnect after reset
                if self.websocket:
                    try:
                        self.websocket.close()
                    except:
                        pass
                
                self._connect()
            else:
                # First reset - just consume initial state
                self._first_reset_done = True
                print("First reset - consuming initial state")
            
            # Receive initial observation and reward
            reward = self.receiveReward()
            observation = self.receiveObservation()
            info = {}
            
            print(f"Reset complete - initial observation shape: {observation.shape}")
            
            return observation, info
        except Exception as e:
            print(f"Error during reset: {e}")
            import traceback
            traceback.print_exc()
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