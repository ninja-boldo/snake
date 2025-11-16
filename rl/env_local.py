import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# Add parent directory to path to import snake_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_py import snake_engine


class SnakeEnv(gym.Env):
    """Custom Snake Environment that follows Gymnasium interface."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, startingBlocks: int = 3, dim: int = 10, distToBorder: int = 3, 
                 render_mode=None):
        """
        Initialize the Snake environment.
        
        Args:
            startingBlocks: Initial length of snake
            dim: Dimension of the game board (dim x dim)
            distToBorder: Minimum distance from border for snake spawn
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        # Store configuration
        self.startingBlocks = startingBlocks
        self.dim = dim
        self.distToBorder = distToBorder
        self.render_mode = render_mode
        
        # Configure snake_engine with our settings
        snake_engine.dimensions = dim
        snake_engine.startBlocks = startingBlocks
        snake_engine.distToBorder = distToBorder
        
        # Define action and observation spaces
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observations: grid with values 0-3
        # 0=empty, 1=body, 2=head, 3=powerup
        self.observation_space = spaces.Box(
            low=0, 
            high=3,
            shape=(dim, dim), 
            dtype=np.int8
        )
        
        # Episode tracking
        self._first_reset_done = False
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0
        self.max_steps = dim * dim * 2  # Prevent infinite episodes
        
        # Game state tracking
        self.last_snake_length = startingBlocks
        self.powerups_collected = 0
    
    def _validate_action(self, action: int, current_dir: int) -> int:
        """
        Validate action to prevent 180-degree turns (snake can't go backwards).
        
        Args:
            action: Proposed action
            current_dir: Current direction
            
        Returns:
            Valid action (either proposed action or continue current direction)
        """
        # Map of opposite directions
        opposite = {0: 2, 1: 3, 2: 0, 3: 1}
        
        # If trying to go opposite direction, continue current direction instead
        if action == opposite.get(current_dir):
            return current_dir
        
        return action
    
    def step(self, action: int):
        """
        Execute one time step within the environment.
        
        Args:
            action: Action to take (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            observation: Current state of the game board
            reward: Reward for this step
            terminated: Whether episode ended due to game over
            truncated: Whether episode ended due to time limit
            info: Additional information dictionary
        """
        if not self._first_reset_done:
            raise RuntimeError("Must call reset() before step()")
        
        self.step_count += 1
        
        try:
            # Validate action (prevent 180-degree turns)
            validated_action = self._validate_action(action, snake_engine.dir)
            
            # Execute step in snake engine
            observation, reward, terminated, info = snake_engine.step(action=validated_action)
            
            # Update tracking
            self.total_reward += reward
            
            # Check for truncation (episode too long)
            truncated = self.step_count >= self.max_steps
            
            # Track snake growth
            current_length = len(snake_engine.bodyElements)
            if current_length > self.last_snake_length:
                self.powerups_collected += 1
                info['powerup_collected'] = True
            else:
                info['powerup_collected'] = False
            self.last_snake_length = current_length
            
            # Add additional info
            info.update({
                'episode': self.episode_count,
                'step': self.step_count,
                'total_reward': self.total_reward,
                'snake_length': current_length,
                'powerups_collected': self.powerups_collected,
                'action_taken': validated_action,
                'action_requested': action,
                'action_was_invalid': validated_action != action
            })
            
            # If episode ended, log results
            if terminated or truncated:
                if self.episode_count % 200 == 0:
                    end_reason = "time limit" if truncated else "collision"
                    print(f"🏁 Episode {self.episode_count} ended ({end_reason})")
                    print(f"   Steps: {self.step_count}")
                    print(f"   Total Reward: {self.total_reward:.2f}")
                    print(f"   Final Snake Length: {current_length}")
                    print(f"   PowerUps Collected: {self.powerups_collected}")
            
            # Spawn new powerup if needed (and game not over)
            if not terminated and len(snake_engine.powerUps) == 0:
                snake_engine.spawnPowerUp()
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"❌ Error in step: {e}")
            import traceback
            traceback.print_exc()
            
            # Return safe defaults on error
            terminated = True
            truncated = False
            observation = np.zeros((self.dim, self.dim), dtype=np.int8)
            reward = snake_engine.lostReward
            info = {'error': str(e), 'episode': self.episode_count}
            
            return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Initial state of the game board
            info: Additional information dictionary
        """
        # Handle seeding
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        # Increment episode counter
        self.episode_count += 1
        self.step_count = 0
        self.total_reward = 0
        self.powerups_collected = 0
        
        if self.episode_count % 200 == 0:  # Don't print on first reset
            print(f"\n{'='*60}")
            print(f"🔄 Starting Episode {self.episode_count}")
            print(f"{'='*60}")
        
        try:
            # Reset the snake engine
            snake_engine.resetGame()
            
            # Initialize new game
            observation = snake_engine.initGame()
            
            # Track initial snake length
            self.last_snake_length = len(snake_engine.bodyElements)
            
            # Build info dict
            info = {
                "episode": self.episode_count,
                "snake_length": self.last_snake_length,
                "powerups_count": len(snake_engine.powerUps)
            }
            
            self._first_reset_done = True
            
            if self.episode_count % 200 == 0 and self.episode_count > 1:
                
                print("✓ Episode initialized")
                print(f"   Snake length: {self.last_snake_length}")
                print(f"   PowerUps: {len(snake_engine.powerUps)}")
                print(f"{'='*60}\n")
            
            return observation, info
            
        except Exception as e:
            print(f"❌ Error during reset: {e}")
            import traceback
            traceback.print_exc()
            
            # Return safe defaults
            observation = np.zeros((self.dim, self.dim), dtype=np.int8)
            info = {"error": str(e), "episode": self.episode_count}
            
            return observation, info
    
    def render(self):
        """
        Render the environment.
        
        For 'human' mode, prints ASCII representation.
        For 'rgb_array' mode, returns RGB array.
        """
        if self.render_mode == "human":
            self._render_ascii()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        return None
    
    def _render_ascii(self):
        """Print ASCII representation of the game state."""
        symbols = {
            0: '·',  # Empty
            1: '○',  # Body
            2: '●',  # Head
            3: '★'   # PowerUp
        }
        
        print("\n┌" + "─" * (self.dim * 2) + "┐")
        for row in snake_engine.worldMap:
            print("│", end="")
            for cell in row:
                print(f"{symbols.get(cell, '?')} ", end="")
            print("│")
        print("└" + "─" * (self.dim * 2) + "┘")
        
        head = snake_engine.getHead()
        print(f"Snake Length: {len(snake_engine.bodyElements)} | "
              f"Head: ({head.x}, {head.y}) | "
              f"PowerUps: {len(snake_engine.powerUps)} | "
              f"Step: {self.step_count}")
    
    def _render_rgb(self) -> np.ndarray:
        """
        Convert game state to RGB image.
        
        Returns:
            RGB array of shape (dim, dim, 3)
        """
        # Define colors (RGB)
        colors = {
            0: [255, 255, 255],  # Empty - white
            1: [0, 255, 0],      # Body - green
            2: [0, 128, 0],      # Head - dark green
            3: [255, 0, 0]       # PowerUp - red
        }
        
        rgb_array = np.zeros((self.dim, self.dim, 3), dtype=np.uint8)
        
        for y in range(self.dim):
            for x in range(self.dim):
                cell_value = snake_engine.worldMap[y][x]
                rgb_array[y, x] = colors.get(int(cell_value), [128, 128, 128])
        
        return rgb_array
    
    def close(self):
        """Clean up resources."""
        snake_engine.resetGame()
        self._first_reset_done = False