from websockets.sync.client import connect
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import json
import time


class SnakeEnv(gym.Env):
    """Custom Snake Environment that follows gym interface."""
    
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
        self.websocket = None
        self._first_reset_done = False
        self.episode_count = 0
        self.step_count = 0
        
        self._connect()
    
    def _connect(self):
        """Establish websocket connection with error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.websocket = connect(self.ws_url, close_timeout=5)
                print(f"✓ Connected to {self.ws_url}")
                # Send ping to identify as Python client
                self.websocket.send("pingfrombackend")
                
                # Wait for ack
                msg = self.websocket.recv(timeout=2)
                if isinstance(msg, bytes):
                    msg = msg.decode('utf-8')
                print(f"← Received: {msg}")
                
                return
            except Exception as e:
                print(f"⚠️ Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise ConnectionError(f"Failed to connect after {max_retries} attempts")
    
    def _recv_message(self, timeout=3.0):  # Reduced from 10s to 3s
        """Receive and decode message properly"""
        try:
            msg = self.websocket.recv(timeout=timeout)
            if isinstance(msg, bytes):
                return msg.decode('utf-8')
            return str(msg)
        except TimeoutError:
            print(f"⚠️ Timeout waiting for message (waited {timeout}s)")
            raise
        except Exception as e:
            print(f"⚠️ Error receiving message: {e}")
            raise
    
    def receiveReward(self, prefix="reward"):
        """Receive and parse reward from websocket"""
        try:
            while True:
                msg_str = self._recv_message()
                
                # Skip ping/ack messages silently
                if msg_str.startswith('ping') or msg_str.startswith('ack'):
                    continue
                    
                if msg_str.startswith(prefix):
                    reward_str = msg_str.removeprefix(prefix)
                    reward = float(reward_str)
                    return reward
                else:
                    print(f"⊘ Skipping non-reward message: {msg_str[:50]}")
            
        except ValueError as e:
            print(f"⚠️ Error parsing reward: {e}")
            return 0.0
        except Exception as e:
            print(f"⚠️ Error receiving reward: {e}")
            raise
    
    def receiveObservation(self, prefix="observation_space"):
        """Receive and parse observation from websocket"""
        try:
            while True:
                msg_str = self._recv_message()
                
                # Skip ping/ack messages silently
                if msg_str.startswith('ping') or msg_str.startswith('ack'):
                    continue
                
                if msg_str.startswith(prefix):
                    obs_str = msg_str.removeprefix(prefix)
                    
                    # Parse JSON observation
                    obs_data = json.loads(obs_str)
                    observation = np.array(obs_data, dtype=np.int8)
                    
                    # Ensure correct shape
                    if observation.shape != (self.dim, self.dim):
                        observation = observation.reshape((self.dim, self.dim))
                    
                    return observation
                else:
                    print(f"⊘ Skipping non-observation: {msg_str[:50]}")
                    
        except Exception as e:
            print(f"⚠️ Error receiving observation: {e}")
            raise
    
    def sendAction(self, action: int, prefix="action"):
        """Send action to websocket"""
        try:
            action_msg = f"{prefix}{str(action)}"
            self.websocket.send(action_msg)
        except Exception as e:
            print(f"⚠️ Error sending action: {e}")
            raise
        
    def step(self, action: int):
        """Execute one time step within the environment"""
        self.step_count += 1
        
        try:
            # Send action
            self.sendAction(action=action)
            
            # Receive reward and observation
            reward = self.receiveReward()
            observation = self.receiveObservation()
            
            # Check termination (game sends -10 or -1000 for game over)
            terminated = (reward <= -10)
            truncated = False
            info = {"step": self.step_count, "episode": self.episode_count}
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"❌ Error in step: {e}")
            # Return safe defaults on error
            terminated = True
            return np.zeros((self.dim, self.dim), dtype=np.int8), -10.0, terminated, False, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        self.episode_count += 1
        self.step_count = 0
        
        try:
            if self._first_reset_done:
                # Subsequent resets - send reset command
                self.websocket.send("reset")
                
                # Game will immediately reset and send state - NO sleep needed!
                # Just wait for the messages to arrive
                
            else:
                # First reset - game already initialized, just consume state
                self._first_reset_done = True
            
            # Receive initial state (game sends immediately after reset)
            reward = self.receiveReward()
            observation = self.receiveObservation()
            
            info = {"episode": self.episode_count}
            
            return observation, info
            
        except Exception as e:
            print(f"❌ Error during reset: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to recover by reconnecting
            print("\n🔄 Attempting to reconnect...")
            try:
                if self.websocket:
                    try:
                        self.websocket.close()
                    except:
                        pass
                
                time.sleep(0.5)  # Reduced from 1s
                self._connect()
                
                # Game should auto-start, try to get initial state
                time.sleep(1)  # Reduced from 2s
                
                reward = self.receiveReward()
                observation = self.receiveObservation()
                
                print("✓ Reconnection successful!")
                return observation, {"episode": self.episode_count}
                
            except Exception as e2:
                print(f"❌ Reconnection failed: {e2}")
                print("⚠️ Returning zero observation")
                return np.zeros((self.dim, self.dim), dtype=np.int8), {}
    
    def close(self):
        """Clean up resources"""
        print("\n🧹 Closing environment...")
        if self.websocket:
            try:
                self.websocket.close()
                print("✓ Websocket connection closed")
            except Exception as e:
                print(f"⚠️ Error closing websocket: {e}")
        super().close()
    
    def render(self):
        """Render the environment (handled by browser)"""
        pass