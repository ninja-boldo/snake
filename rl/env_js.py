from websockets.sync.client import connect
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import json
import time

class SnakeEnv(gym.Env):
    """WebSocket-based Snake Environment that works with existing JS implementation."""
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, startingBlocks: int = 3, dim: int = 10, distToBorder: int = 3, 
                 ws_url: str = "ws://localhost:3030", use_features: bool = False):
        super().__init__()
        self.startingBlocks = startingBlocks
        self.dim = dim
        self.distToBorder = distToBorder
        self.use_features = use_features
        self.action_space = spaces.Discrete(4)
        
        # Set observation space based on feature usage
        if use_features:
            self.observation_space = spaces.Box(
                low=-1.0, 
                high=1.0,
                shape=(19,), 
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0, 
                high=3,
                shape=(dim, dim), 
                dtype=np.int8
            )
        
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
    
    def _recv_message(self, timeout=3.0):
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
    
    def _infer_direction(self, grid):
        """
        Infer snake direction from grid by examining head and next body segment
        Returns: 0=up, 1=right, 2=down, 3=left, or 0 (default) if undetermined
        """
        head_pos = None
        body_segments = []
        
        # Find head (2) and all body segments (1)
        for y in range(self.dim):
            for x in range(self.dim):
                cell = grid[y][x]
                if cell == 2:  # Head
                    head_pos = (x, y)
                elif cell == 1:  # Body
                    body_segments.append((x, y))
        
        # If we don't have a head, default to up
        if head_pos is None:
            return 0
        
        head_x, head_y = head_pos
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        direction_names = ["up", "right", "down", "left"]
        
        # Check which direction has the next body segment
        for i, (dx, dy) in enumerate(directions):
            nx, ny = head_x + dx, head_y + dy
            if 0 <= nx < self.dim and 0 <= ny < self.dim and grid[ny][nx] == 1:
                return i
        
        # If no body segment found (just the head), use previous direction logic
        # Check all adjacent cells for body parts
        adjacent_body = []
        for i, (dx, dy) in enumerate(directions):
            nx, ny = head_x + dx, head_y + dy
            if 0 <= nx < self.dim and 0 <= ny < self.dim and grid[ny][nx] in [1, 2]:
                adjacent_body.append((i, (nx, ny)))
        
        # If exactly one adjacent body part, use it to determine direction
        if len(adjacent_body) == 1:
            return adjacent_body[0][0]
        
        # Default to up if we can't determine direction
        return 0
    
    def _extract_features(self, grid):
        """
        Convert grid observation into 19-dimensional feature vector
        Works with existing JS implementation that only sends grid data
        """
        # Find all relevant positions from grid
        head = None
        body_positions = []
        powerUps = []
        
        for y in range(self.dim):
            for x in range(self.dim):
                cell = grid[y][x]
                if cell == 2:  # Head
                    head = (x, y)
                elif cell == 1:  # Body
                    body_positions.append((x, y))
                elif cell == 3:  # Power-up
                    powerUps.append((x, y))
        
        # Fallback if head not found (shouldn't happen)
        if head is None:
            head = (0, 0)
            print("⚠️ Head not found in grid - using (0,0) as fallback")
        
        # Infer direction from grid
        direction = self._infer_direction(grid)
        
        # Find tail position (end of snake chain)
        snake_positions = set(body_positions)
        if head:
            snake_positions.add(head)
        
        # Function to count adjacent snake segments
        def count_adjacent(pos):
            x, y = pos
            count = 0
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.dim and 0 <= ny < self.dim:
                    if (nx, ny) in snake_positions:
                        count += 1
            return count
        
        # Find all end points (have only 1 neighbor)
        ends = [pos for pos in snake_positions if count_adjacent(pos) == 1]
        
        # Determine tail (the end that's not the head)
        tail = head  # Default for single-segment snake
        if ends:
            if head in ends and len(ends) > 1:
                ends.remove(head)
                tail = ends[0]
            elif len(ends) == 1:
                tail = ends[0]

        # Start building features
        features = []
        
        # 1. Current direction (one-hot encoding)
        dir_encoding = [0, 0, 0, 0]
        dir_encoding[direction] = 1
        features.extend(dir_encoding)
        
        # 2. Danger in each direction (4 values)
        head_x, head_y = head
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        for dx, dy in directions:
            nx, ny = head_x + dx, head_y + dy
            danger = 0
            # Check boundaries
            if nx < 0 or nx >= self.dim or ny < 0 or ny >= self.dim:
                danger = 1
            # Check body collision
            elif 0 <= ny < self.dim and 0 <= nx < self.dim and grid[ny][nx] in [1, 2]:
                danger = 1
            features.append(danger)
        
        # 3. Food direction and relative position (6 values)
        if powerUps:
            food_x, food_y = powerUps[0]
            food_dx = food_x - head_x
            food_dy = food_y - head_y
            
            # Food direction (one-hot)
            food_dir = [0, 0, 0, 0]
            if abs(food_dy) > abs(food_dx):
                food_dir[0 if food_dy < 0 else 2] = 1  # Up/Down
            else:
                food_dir[1 if food_dx > 0 else 3] = 1   # Right/Left
            features.extend(food_dir)
            
            # Normalized relative position
            features.append(food_dx / self.dim)
            features.append(food_dy / self.dim)
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # 4. Snake length (normalized)
        features.append(len(snake_positions) / (self.dim * self.dim))
        
        # 5. Head position (normalized)
        features.append(head_x / self.dim)
        features.append(head_y / self.dim)
        
        # 6. Tail position (normalized)
        if len(snake_positions) > 1:
            tail_x, tail_y = tail
            features.append(tail_x / self.dim)
            features.append(tail_y / self.dim)
        else:
            features.extend([0, 0])
        
        return np.array(features, dtype=np.float32)

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
                    try:
                        obs_data = json.loads(obs_str)
                    except json.JSONDecodeError:
                        print(f"⚠️ Failed to parse JSON: {obs_str[:50]}")
                        raise
                    
                    # Convert to numpy array and ensure correct shape
                    observation = np.array(obs_data, dtype=np.int8)
                    if observation.shape != (self.dim, self.dim):
                        observation = observation.reshape((self.dim, self.dim))
                    
                    # Process based on feature usage
                    if self.use_features:
                        observation = self._extract_features(observation)
                    
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
            default_shape = (19,) if self.use_features else (self.dim, self.dim)
            dtype = np.float32 if self.use_features else np.int8
            return np.zeros(default_shape, dtype=dtype), -10.0, terminated, False, {}
    
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
            else:
                # First reset - game already initialized
                self._first_reset_done = True
            
            # Receive initial state
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
                
                time.sleep(0.5)
                self._connect()
                
                # Try to get initial state
                time.sleep(1)
                
                reward = self.receiveReward()
                observation = self.receiveObservation()
                
                print("✓ Reconnection successful!")
                return observation, {"episode": self.episode_count}
                
            except Exception as e2:
                print(f"❌ Reconnection failed: {e2}")
                print("⚠️ Returning default observation")
                default_shape = (19,) if self.use_features else (self.dim, self.dim)
                dtype = np.float32 if self.use_features else np.int8
                return np.zeros(default_shape, dtype=dtype), {}
    
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