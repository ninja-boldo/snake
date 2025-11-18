import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_py import snake_engine


class RewardShaper:
    def __init__(self, dim: int):
        self.dim = dim
        self.prev_distance = None
        self.survival_steps = 0
        
    def reset(self):
        self.prev_distance = None
        self.survival_steps = 0
    
    def calculate(self, worldMap, bodyElements, powerUps, ate_powerup, collision):
        reward = 0.0
        
        if collision:
            return -10.0
        
        if ate_powerup:
            reward += 10.0
            self.prev_distance = None
        
        if powerUps and bodyElements:
            head = bodyElements[0]
            food = powerUps[0]
            current_distance = abs(head.x - food.x) + abs(head.y - food.y)
            
            if self.prev_distance is not None:
                if current_distance < self.prev_distance:
                    reward += 0.1
                elif current_distance > self.prev_distance:
                    reward -= 0.1
            
            self.prev_distance = current_distance
        
        self.survival_steps += 1
        reward += 0.01
        
        snake_length = len(bodyElements)
        reward += 0.01 * snake_length
        
        if bodyElements:
            head = bodyElements[0]
            danger_count = 0
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = head.x + dx, head.y + dy
                if nx < 0 or nx >= self.dim or ny < 0 or ny >= self.dim:
                    danger_count += 1
                elif 0 <= ny < self.dim and 0 <= nx < self.dim:
                    if worldMap[ny][nx] in [1, 2]:
                        danger_count += 1
            
            if danger_count >= 3:
                reward -= 0.05
        
        return reward


def extract_features(worldMap, bodyElements, powerUps, current_dir, dim):
    head = bodyElements[0]
    features = []
    
    dir_encoding = [0, 0, 0, 0]
    dir_encoding[current_dir] = 1
    features.extend(dir_encoding)
    
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    for dx, dy in directions:
        new_x, new_y = head.x + dx, head.y + dy
        is_danger = 0
        if new_x < 0 or new_x >= dim or new_y < 0 or new_y >= dim:
            is_danger = 1
        elif 0 <= new_y < dim and 0 <= new_x < dim:
            if worldMap[new_y][new_x] in [1, 2]:
                is_danger = 1
        features.append(is_danger)
    
    if powerUps:
        food = powerUps[0]
        food_dx = food.x - head.x
        food_dy = food.y - head.y
        
        food_dir = [0, 0, 0, 0]
        if abs(food_dy) > abs(food_dx):
            food_dir[0 if food_dy < 0 else 2] = 1
        else:
            food_dir[1 if food_dx > 0 else 3] = 1
        features.extend(food_dir)
        
        features.append(food_dx / dim)
        features.append(food_dy / dim)
    else:
        features.extend([0, 0, 0, 0, 0, 0])
    
    features.append(len(bodyElements) / (dim * dim))
    features.append(head.x / dim)
    features.append(head.y / dim)
    
    if len(bodyElements) > 1:
        tail = bodyElements[-1]
        features.append(tail.x / dim)
        features.append(tail.y / dim)
    else:
        features.extend([0, 0])
    
    return np.array(features, dtype=np.float32)


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, startingBlocks: int = 3, dim: int = 10, distToBorder: int = 3, 
                 render_mode=None, use_features=True):
        super().__init__()
        
        self.startingBlocks = startingBlocks
        self.dim = dim
        self.distToBorder = distToBorder
        self.render_mode = render_mode
        self.use_features = use_features
        
        snake_engine.dimensions = dim
        snake_engine.startBlocks = startingBlocks
        snake_engine.distToBorder = distToBorder
        
        self.action_space = spaces.Discrete(4)
        
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
        
        self._first_reset_done = False
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0
        self.max_steps = dim * dim * 2
        
        self.last_snake_length = startingBlocks
        self.powerups_collected = 0
        self.reward_shaper = RewardShaper(dim)
    
    def _validate_action(self, action: int, current_dir: int) -> int:
        opposite = {0: 2, 1: 3, 2: 0, 3: 1}
        if action == opposite.get(current_dir):
            return current_dir
        return action
    
    def _get_observation(self):
        if self.use_features:
            return extract_features(
                snake_engine.worldMap,
                snake_engine.bodyElements,
                snake_engine.powerUps,
                snake_engine.dir,
                self.dim
            )
        else:
            print(f"snake_engine.worldMap.copy(): {snake_engine.worldMap.copy()}")
            return snake_engine.worldMap.copy()
    
    def step(self, action: int):
        if not self._first_reset_done:
            raise RuntimeError("Must call reset() before step()")
        
        self.step_count += 1
        
        try:
            validated_action = self._validate_action(action, snake_engine.dir)
            prev_length = len(snake_engine.bodyElements)
            
            _, _, terminated, info = snake_engine.step(action=validated_action)
            
            ate_powerup = len(snake_engine.bodyElements) > prev_length
            
            reward = self.reward_shaper.calculate(
                worldMap=snake_engine.worldMap,
                bodyElements=snake_engine.bodyElements,
                powerUps=snake_engine.powerUps,
                ate_powerup=ate_powerup,
                collision=terminated
            )
            
            observation = self._get_observation()
            self.total_reward += reward
            truncated = self.step_count >= self.max_steps
            
            current_length = len(snake_engine.bodyElements)
            if current_length > self.last_snake_length:
                self.powerups_collected += 1
            self.last_snake_length = current_length
            
            info.update({
                'episode': self.episode_count,
                'step': self.step_count,
                'total_reward': self.total_reward,
                'snake_length': current_length,
                'powerups_collected': self.powerups_collected,
                'action_taken': validated_action,
            })
            
            if terminated or truncated:
                if self.episode_count % 100 == 0:
                    end_reason = "time" if truncated else "collision"
                    print(f"Ep {self.episode_count} ended ({end_reason}) | "
                          f"Steps: {self.step_count} | Reward: {self.total_reward:.2f} | "
                          f"Length: {current_length} | PowerUps: {self.powerups_collected}")
            
            if not terminated and len(snake_engine.powerUps) == 0:
                snake_engine.spawnPowerUp()
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error in step: {e}")
            import traceback
            traceback.print_exc()
            
            terminated = True
            truncated = False
            observation = self._get_observation()
            reward = -10.0
            info = {'error': str(e), 'episode': self.episode_count}
            
            return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        self.episode_count += 1
        self.step_count = 0
        self.total_reward = 0
        self.powerups_collected = 0
        self.reward_shaper.reset()
        
        try:
            snake_engine.resetGame()
            snake_engine.initGame()
            
            self.last_snake_length = len(snake_engine.bodyElements)
            observation = self._get_observation()
            
            info = {
                "episode": self.episode_count,
                "snake_length": self.last_snake_length,
                "powerups_count": len(snake_engine.powerUps)
            }
            
            self._first_reset_done = True
            
            return observation, info
            
        except Exception as e:
            print(f"Error during reset: {e}")
            import traceback
            traceback.print_exc()
            
            observation = self._get_observation()
            info = {"error": str(e), "episode": self.episode_count}
            
            return observation, info
    
    def render(self):
        if self.render_mode == "human":
            self._render_ascii()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        return None
    
    def _render_ascii(self):
        symbols = {0: '·', 1: '○', 2: '●', 3: '★'}
        
        print("\n┌" + "─" * (self.dim * 2) + "┐")
        for row in snake_engine.worldMap:
            print("│", end="")
            for cell in row:
                print(f"{symbols.get(cell, '?')} ", end="")
            print("│")
        print("└" + "─" * (self.dim * 2) + "┘")
        
        head = snake_engine.getHead()
        print(f"Length: {len(snake_engine.bodyElements)} | "
              f"Head: ({head.x}, {head.y}) | "
              f"PowerUps: {len(snake_engine.powerUps)} | "
              f"Step: {self.step_count}")
    
    def _render_rgb(self) -> np.ndarray:
        colors = {
            0: [255, 255, 255],
            1: [0, 255, 0],
            2: [0, 128, 0],
            3: [255, 0, 0]
        }
        
        rgb_array = np.zeros((self.dim, self.dim, 3), dtype=np.uint8)
        
        for y in range(self.dim):
            for x in range(self.dim):
                cell_value = snake_engine.worldMap[y][x]
                rgb_array[y, x] = colors.get(int(cell_value), [128, 128, 128])
        
        return rgb_array
    
    def close(self):
        snake_engine.resetGame()
        self._first_reset_done = False