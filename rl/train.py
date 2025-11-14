from env import CustomEnv
import time

print("="*50)
print("Starting Snake RL Training")
print("="*50)

# Create environment
print("\n1. Creating environment...")
env = CustomEnv(startingBlocks=3, dim=20, distToBorder=3)

try:
    # Reset environment
    print("\n2. Resetting environment...")
    observation, info = env.reset(seed=42)
    print(f"✓ Initial observation shape: {observation.shape}")
    print(f"✓ Initial observation:\n{observation}")
    
    # Run some episodes
    for episode in range(3):
        print(f"\n{'='*50}")
        print(f"EPISODE {episode + 1}")
        print(f"{'='*50}")
        
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(100000):  # Max 100 steps per episode
            # Sample random action
            action = env.action_space.sample()
            print(f"\n--- Step {step + 1} ---")
            print(f"Taking action: {action} (0=up, 1=right, 2=down, 3=left)")
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Reward: {reward:.2f}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            
            if terminated or truncated:
                print(f"\n🏁 Episode ended after {step + 1} steps")
                break
            
            time.sleep(0.3)  # Slow down for visibility
        
        print(f"\n📊 Episode {episode + 1} Summary:")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Steps: {step + 1}")

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
except Exception as e:
    print(f"\n\n❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    env.close()
    print("\n✓ Environment closed")
    print("="*50)