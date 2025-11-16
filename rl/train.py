import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from typing import Optional, Tuple

# Import the fixed Gymnasium environment
from env_local import SnakeEnv

try:
    torch.set_float32_matmul_precision("medium")
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True


class DQNetwork(nn.Module):
    """Deep Q-Network with improved architecture"""
    def __init__(self, input_size, output_size, hidden_size=256):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),  # Prevent overfitting
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Fast replay buffer backed by contiguous numpy arrays."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.idx = 0
        self.full = False

    def add(self, state, action, reward, next_state, done):
        flat_state = np.asarray(state, dtype=np.float32).reshape(-1)
        flat_next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self.states[self.idx] = flat_state
        self.next_states[self.idx] = flat_next_state
        self.rewards[self.idx] = reward
        self.actions[self.idx] = action
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def can_sample(self, batch_size: int) -> bool:
        return len(self) >= batch_size

    def sample(self, batch_size: int):
        max_index = self.capacity if self.full else self.idx
        indices = np.random.choice(max_index, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


class DQNAgent:
    """Deep Q-Network agent with target network and fast replay buffer."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.0005,
        gamma=0.99,
        use_target_network=True,
        buffer_capacity: int = 100_000,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        compile_model: bool = False,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.use_target_network = use_target_network
        self.update_target_every = 100
        self.step_count = 0
        self.device = device or torch.device("cpu")
        self.amp_enabled = use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self.memory = ReplayBuffer(buffer_capacity, state_size)

        self.model = DQNetwork(state_size, action_size).to(self.device)
        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        if self.use_target_network:
            self.target_model = DQNetwork(state_size, action_size).to(self.device)
            self.update_target_network()
            if compile_model and hasattr(torch, "compile"):
                self.target_model = torch.compile(self.target_model)  # type: ignore[attr-defined]
        else:
            self.target_model = None

    def update_target_network(self):
        if self.use_target_network and self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, inference_mode: bool = False):
        if not inference_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def replay(self, batch_size=32):
        if not self.memory.can_sample(batch_size):
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.float32)).to(self.device)

        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            current_q_values = self.model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            if self.use_target_network and self.target_model is not None:
                next_q_values = self.target_model(next_states_t).max(1)[0]
            else:
                next_q_values = self.model(next_states_t).max(1)[0]

            target_q_values = rewards_t + (1 - dones_t) * self.gamma * next_q_values
            loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad(set_to_none=True)

        if self.amp_enabled:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()

        self.step_count += 1
        if self.use_target_network and self.step_count % self.update_target_every == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def save(self, filepath="snake_dqn_model.pth"):
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "step_count": self.step_count,
        }

        if self.use_target_network and self.target_model is not None:
            save_dict["target_model_state_dict"] = self.target_model.state_dict()

        torch.save(save_dict, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath="snake_dqn_model.pth"):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint["epsilon"]
            self.step_count = checkpoint.get("step_count", 0)

            if self.use_target_network and self.target_model is not None and "target_model_state_dict" in checkpoint:
                self.target_model.load_state_dict(checkpoint["target_model_state_dict"])

            print(f"✓ Model loaded from {filepath}")
            print(f"  Epsilon: {self.epsilon:.4f}, Steps: {self.step_count}")
            return True

        print(f"⚠️  No saved model found at {filepath}")
        return False


def train_agent(env, agent, episodes=1000, max_steps=500, save_every=1000, 
                model_path="snake_dqn_model.pth", render_every=0):
    """
    Training loop with improved tracking
    
    Args:
        env: Gymnasium environment
        agent: DQN agent
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_every: Save model every N episodes
        model_path: Path to save model
        render_every: Render every N episodes (0 = never)
    """
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    best_reward = float('-inf')
    
    print(f"\n{'='*60}")
    print(f"Training for {episodes} episodes")
    print(f"{'='*60}\n")
    
    for episode in range(episodes):
        obs, info = env.reset()
        state = obs.flatten()
        total_reward = 0
        episode_loss = 0
        loss_count = 0
        
        # Render if requested
        should_render = render_every > 0 and (episode + 1) % render_every == 0
        
        for step in range(max_steps):
            # Agent chooses action
            action = agent.act(state)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = obs.flatten()
            total_reward += reward
            
            # Store experience
            agent.remember(state, action, reward, next_state, terminated or truncated)
            
            # Train agent
            loss = agent.replay(batch_size=32)
            if loss > 0:
                episode_loss += loss
                loss_count += 1
            
            state = next_state
            
            if should_render:
                env.render()
                time.sleep(0.05)
            
            if terminated or truncated:
                break
        
        # Track metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        episode_losses.append(avg_loss)
        
        # Update best reward
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(model_path.replace('.pth', '_best.pth'))
        
        # Logging
        if (episode + 1) % 200 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_lengths = episode_lengths[-10:]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            max_length = max(recent_lengths)
            
            print(f"Ep {episode + 1:4d}/{episodes} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Length: {avg_length:5.1f} (max: {max_length:3d}) | "
                  f"Loss: {avg_loss:6.4f} | "
                  f"ε: {agent.epsilon:.3f}")
        
        # Save periodically
        if (episode + 1) % save_every == 0:
            agent.save(model_path)
            print(f"  → Checkpoint saved (Episode {episode + 1})")
    
    return episode_rewards, episode_lengths, episode_losses


def evaluate_agent(env, agent, num_episodes=10, max_steps=500, render=True):
    """Evaluate the agent without exploration"""
    print(f"\n{'='*60}")
    print(f"EVALUATION MODE ({num_episodes} episodes)")
    print(f"{'='*60}\n")
    
    total_rewards = []
    total_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = obs.flatten()
        total_reward = 0
        
        for step in range(max_steps):
            # Act greedily (no exploration)
            action = agent.act(state, inference_mode=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = obs.flatten()
            total_reward += reward
            state = next_state
            
            if render:
                env.render()
                time.sleep(0.05)
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        total_lengths.append(step + 1)
        
        print(f"Episode {episode + 1:2d} | "
              f"Reward: {total_reward:7.2f} | "
              f"Length: {step + 1:3d} | "
              f"Snake Length: {info['snake_length']}")
    
    print(f"\n{'='*60}")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Length: {np.mean(total_lengths):.1f} ± {np.std(total_lengths):.1f}")
    print(f"Best Reward: {max(total_rewards):.2f}")
    print(f"{'='*60}\n")


def plot_training_results(rewards, lengths, losses, save_path="training_results.png"):
    """Plot comprehensive training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(rewards, alpha=0.3, label='Episode Reward')
    window = min(100, len(rewards) // 10)
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 
                       label=f'{window}-Episode Moving Avg', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    axes[0, 1].plot(lengths, alpha=0.3, label='Episode Length')
    if len(lengths) >= window:
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(lengths)), moving_avg, 
                       label=f'{window}-Episode Moving Avg', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training Loss
    axes[1, 0].plot(losses, alpha=0.5, label='Loss')
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(losses)), moving_avg, 
                       label=f'{window}-Episode Moving Avg', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward Distribution
    axes[1, 1].hist(rewards, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(rewards), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(rewards):.2f}')
    axes[1, 1].set_xlabel('Total Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training plots saved to {save_path}")
    plt.close()


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("Snake RL Training with DQN")
    print("="*60)
    
    # Configuration
    CONFIG = {
        'train_mode': True,        # Set to False for evaluation only
        'load_model': False,       # Load existing model
        'model_path': 'snake_dqn.pth',
        'episodes': 50000,          # Training episodes
        'max_steps': 500,          # Max steps per episode
        'save_every': 1000,         # Save every N episodes
        'render_every': 0,         # Render every N episodes (0=never)
        'eval_episodes': 10,       # Episodes for evaluation
        'dim': 20,                 # Board size
        'starting_blocks': 3,      # Initial snake length
        'learning_rate': 0.001,
        'gamma': 0.99,
        'use_target_network': True
    }
    
    # Create environment
    print("\n1. Creating environment...")
    env = SnakeEnv(
        startingBlocks=CONFIG['starting_blocks'],
        dim=CONFIG['dim'],
        distToBorder=3,
        render_mode="human" if CONFIG['render_every'] > 0 else None
    )
    
    # Get dimensions
    obs, _ = env.reset()
    state_size = obs.flatten().shape[0]
    action_size = env.action_space.n
    
    print(f"✓ State size: {state_size} ({CONFIG['dim']}x{CONFIG['dim']})")
    print(f"✓ Action size: {action_size}")
    
    # Create agent
    print("\n2. Creating DQN agent...")
    agent = DQNAgent(
        state_size,
        action_size,
        learning_rate=CONFIG['learning_rate'],
        gamma=CONFIG['gamma'],
        use_target_network=CONFIG['use_target_network']
    )
    print(f"✓ Learning rate: {CONFIG['learning_rate']}")
    print(f"✓ Gamma: {CONFIG['gamma']}")
    print(f"✓ Target network: {CONFIG['use_target_network']}")
    
    # Load existing model if requested
    if CONFIG['load_model']:
        agent.load(CONFIG['model_path'])
    
    try:
        if CONFIG['train_mode']:
            # Training
            print("\n3. Starting training...")
            rewards, lengths, losses = train_agent(
                env, agent,
                episodes=CONFIG['episodes'],
                max_steps=CONFIG['max_steps'],
                save_every=CONFIG['save_every'],
                model_path=CONFIG['model_path'],
                render_every=CONFIG['render_every']
            )
            
            # Final save
            agent.save(CONFIG['model_path'])
            
            # Plot results
            print("\n4. Generating plots...")
            plot_training_results(rewards, lengths, losses)
            
            # Evaluate
            print("\n5. Final evaluation...")
            evaluate_agent(env, agent, num_episodes=CONFIG['eval_episodes'], 
                         render=False)
        else:
            # Evaluation only
            if not agent.load(CONFIG['model_path']):
                print("❌ Cannot evaluate without a trained model!")
            else:
                evaluate_agent(env, agent, num_episodes=CONFIG['eval_episodes'],
                             render=True)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        agent.save(CONFIG['model_path'])
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        agent.save(CONFIG['model_path'].replace('.pth', '_error.pth'))
    finally:
        env.close()
        print("\n✓ Training complete!")
        print("="*60)