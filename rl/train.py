from env import CustomEnv, ShiftWrapper
import time

# Create environment
env = CustomEnv(startingBlocks=3, dim=10, distToBorder=3)

try:
    # Reset environment
    print("Resetting environment...")
    observation, info = env.reset(seed=42)
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial observation:\n{observation}")
    
    # Run some episodes
    for episode in range(3):
        print(f"\n=== Episode {episode + 1} ===")
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(10000):  # Max 20 steps per episode
            # Sample random action
            action = env.action_space.sample()
            print(f"\nStep {step + 1}: Taking action {action}")
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Reward: {reward}, Total: {total_reward}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            
            if terminated or truncated:
                print(f"Episode ended after {step + 1} steps")
                break
            
            time.sleep(0.5)  # Slow down for visibility
        
        print(f"Episode {episode + 1} total reward: {total_reward}")

finally:
    env.close()
    print("\nEnvironment closed")