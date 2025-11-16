from env import CustomEnv
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os


class SimpleQNetwork(nn.Module):
    """Lightweight neural network for Q-learning"""
    def __init__(self, input_size, output_size):
        super(SimpleQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Lightweight DQN agent for reinforcement learning"""
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # Q-Network
        self.model = SimpleQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, inference_mode=False):
        """Choose action using epsilon-greedy policy"""
        if not inference_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()
    
    def replay(self, batch_size=32):
        """Train on a random batch from memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * self.model(next_state_tensor).max().item()
            
            current_q = self.model(state_tensor)
            target_q = current_q.clone()
            target_q[0][action] = target
            
            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath="snake_dqn_model.pth"):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath="snake_dqn_model.pth"):
        """Load a trained model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"✓ Model loaded from {filepath}")
            return True
        else:
            print(f"⚠️  No saved model found at {filepath}")
            return False


def train_agent(env, agent, episodes=50000, save_every=1000, model_path="snake_dqn_model.pth"):
    """Training loop"""
    episode_rewards = []
    episode_numbers = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        state = obs.flatten()
        total_reward = 0
        
        for step in range(1000):
            # Agent chooses action
            action = agent.act(state)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = obs.flatten()
            total_reward += reward
            
            # Store experience
            agent.remember(state, action, reward, next_state, terminated or truncated)
            
            state = next_state
            time.sleep(0.06)
            
            if terminated or truncated:
                break
        
        # Train agent
        agent.replay()
        
        episode_rewards.append(total_reward)
        episode_numbers.append(episode + 1)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Steps: {step + 1}")
        
        # Save periodically
        if (episode + 1) % save_every == 0:
            agent.save(model_path)
    
    return episode_numbers, episode_rewards


def inference_mode(env, agent, num_episodes=10, render_delay=0.05):
    """Run the agent in inference mode (no training)"""
    print("\n" + "="*50)
    print("INFERENCE MODE")
    print("="*50)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = obs.flatten()
        total_reward = 0
        
        print(f"\n--- Inference Episode {episode + 1} ---")
        
        for step in range(1000):
            action = agent.act(state, inference_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = obs.flatten()
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
            
            time.sleep(render_delay)  # Only sleep in inference for visualization
        
        print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f} | Steps: {step + 1}")


# Main execution
if __name__ == "__main__":
    print("="*50)
    print("Starting Snake RL Training with DQN")
    print("="*50)
    
    # Configuration
    TRAIN_MODE = True  # Set to False for inference only
    LOAD_MODEL = False  # Set to True to continue training from saved model
    MODEL_PATH = "snake_dqn_model.pth"
    
    # Create environment
    print("\n1. Creating environment...")
    env = CustomEnv(startingBlocks=3, dim=20, distToBorder=3)
    
    # Get state and action dimensions
    obs, _ = env.reset()
    state_size = obs.flatten().shape[0]
    action_size = env.action_space.n
    
    print(f"✓ State size: {state_size}")
    print(f"✓ Action size: {action_size}")
    
    # Create agent
    print("\n2. Creating DQN agent...")
    agent = DQNAgent(state_size, action_size)
    
    # Load existing model if requested
    if LOAD_MODEL:
        agent.load(MODEL_PATH)
    
    try:
        if TRAIN_MODE:
            # Training
            print("\n3. Starting training...")
            episodes, rewards = train_agent(env, agent, episodes=15, model_path=MODEL_PATH)
            
            # Final save
            agent.save(MODEL_PATH)
            
            # Plot results
            print("\n4. Plotting results...")
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, rewards, alpha=0.3, label='Episode Reward')
            
            # Moving average
            window = 100
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(range(window, len(rewards) + 1), moving_avg, label=f'{window}-Episode Moving Avg')
            
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Snake DQN Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig("result_fig.png", dpi=150, bbox_inches='tight')
            print("✓ Results saved to result_fig.png")
        else:
            # Inference only
            if not agent.load(MODEL_PATH):
                print("❌ Cannot run inference without a trained model!")
            else:
                inference_mode(env, agent, num_episodes=10)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        agent.save(MODEL_PATH)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n✓ Environment closed")
        print("="*50)