import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from typing import Optional

from env_js import SnakeEnv

try:
    torch.set_float32_matmul_precision("medium")
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True


class DQNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256):
        print(f"input_size: {input_size}, output_size: {output_size}, hidden_size: {hidden_size}")
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                # small init for final layer
                if m.out_features == self.network[-1].out_features:
                    nn.init.uniform_(m.weight, -3e-3, 3e-3)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x.float())


class ReplayBuffer:
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


def resolve_device(requested: str = "cpu") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_epsilon(episode: int, start: float, end: float, decay_episodes: int) -> float:
    if episode >= decay_episodes:
        return end
    progress = episode / decay_episodes
    return start - progress * (start - end)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_episodes=2000,
        use_target_network=True,
        target_update_freq=500,
        buffer_capacity: int = 50000,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        compile_model: bool = False,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.learning_rate = learning_rate
        self.use_target_network = use_target_network
        self.target_update_freq = target_update_freq
        self.step_count = 0
        self.episode_count = 0
        self.device = device or torch.device("cpu")
        self.amp_enabled = use_amp and self.device.type == "cuda"
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self.memory = ReplayBuffer(buffer_capacity, state_size)

        self.model = DQNetwork(state_size, action_size).to(self.device)
        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        if self.use_target_network:
            self.target_model = DQNetwork(state_size, action_size).to(self.device)
            self.update_target_network()
            if compile_model and hasattr(torch, "compile"):
                self.target_model = torch.compile(self.target_model)
        else:
            self.target_model = None

    def update_target_network(self):
        if self.use_target_network and self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        self.episode_count += 1
        self.epsilon = get_epsilon(
            self.episode_count, 
            self.epsilon_start, 
            self.epsilon_end, 
            self.epsilon_decay_episodes
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, inference_mode: bool = False):
        if not inference_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def replay(self, batch_size=64):
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
        if self.use_target_network and self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def save(self, filepath="snake_dqn_model.pth"):
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "step_count": self.step_count,
        }

        if self.use_target_network and self.target_model is not None:
            save_dict["target_model_state_dict"] = self.target_model.state_dict()

        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath="snake_dqn_model.pth"):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint["epsilon"]
            self.episode_count = checkpoint.get("episode_count", 0)
            self.step_count = checkpoint.get("step_count", 0)

            if self.use_target_network and self.target_model is not None and "target_model_state_dict" in checkpoint:
                self.target_model.load_state_dict(checkpoint["target_model_state_dict"])

            print(f"Model loaded from {filepath}")
            print(f"  Epsilon: {self.epsilon:.4f}, Episode: {self.episode_count}, Steps: {self.step_count}")
            return True

        print(f"No saved model found at {filepath}")
        return False


def train_agent(env, agent, episodes=20000, max_steps=1000, save_every=500, 
                model_path="snake_dqn.pth", render_every=0, min_buffer=1000, 
                batch_size=64, patience=2000):
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    best_avg_reward = float('-inf')
    episodes_without_improvement = 0
    
    print(f"\nTraining for {episodes} episodes")
    print(f"Device: {agent.device}, Batch size: {batch_size}, Min buffer: {min_buffer}\n")
    
    for episode in range(episodes):
        obs, info = env.reset()
        state = obs.astype(np.float32).flatten()
        total_reward = 0
        episode_loss = 0
        loss_count = 0
        
        should_render = render_every > 0 and (episode + 1) % render_every == 0
        
        for step in range(max_steps):
            action = agent.act(state)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = obs.astype(np.float32).flatten()
            total_reward += reward
            
            agent.remember(state, action, reward, next_state, terminated or truncated)
            
            if len(agent.memory) >= min_buffer:
                loss = agent.replay(batch_size=batch_size)
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1
            
            state = next_state
            
            if should_render:
                env.render()
                time.sleep(0.05)
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        episode_losses.append(avg_loss)
        
        agent.update_epsilon()
        
        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            recent_lengths = episode_lengths[-100:]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            max_length = max(recent_lengths)
            
            print(f"Ep {episode + 1:5d}/{episodes} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Length: {avg_length:5.1f} (max: {max_length:3d}) | "
                  f"Loss: {avg_loss:6.4f} | "
                  f"ε: {agent.epsilon:.3f}")
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                episodes_without_improvement = 0
                agent.save(model_path.replace('.pth', '_best.pth'))
            else:
                episodes_without_improvement += 100
            
            if episodes_without_improvement >= patience:
                print(f"\nEarly stopping: No improvement for {patience} episodes")
                break
        
        if (episode + 1) % save_every == 0:
            agent.save(model_path)
    
    agent.save(model_path)
    return episode_rewards, episode_lengths, episode_losses


def evaluate_agent(env, agent, num_episodes=20, max_steps=1000, render=False):
    print(f"\nEVALUATION MODE ({num_episodes} episodes)\n")
    
    total_rewards = []
    total_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = obs.astype(np.float32).flatten()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state, inference_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = obs.astype(np.float32).flatten()
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
              f"Length: {step + 1:3d}")
    
    print(f"\nAverage Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Length: {np.mean(total_lengths):.1f} ± {np.std(total_lengths):.1f}")
    print(f"Best Reward: {max(total_rewards):.2f}\n")


def plot_training_results(rewards, lengths, losses, save_path="training_results.png"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(rewards, alpha=0.3, label='Episode Reward')
    window = max(1, min(100, len(rewards) // 10 or 1))
    if len(rewards) >= window and window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 
                       label=f'{window}-Ep MA', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(lengths, alpha=0.3, label='Episode Length')
    if len(lengths) >= window and window > 1:
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(lengths)), moving_avg, 
                       label=f'{window}-Ep MA', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(losses, alpha=0.5, label='Loss')
    if len(losses) >= window and window > 1:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(losses)), moving_avg, 
                       label=f'{window}-Ep MA', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
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
    print(f"Training plots saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    CONFIG = {
        'episodes': 20000,
        'max_steps': 1000,
        'save_every': 500,
        'render_every': 0,
        'eval_episodes': 20,
        'dim': 10,  # FIXED: Changed from 10 to 20 to match the game
        'starting_blocks': 3,
        'learning_rate': 0.0003,
        'gamma': 0.95,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_episodes': 5000,
        'target_update_freq': 500,
        'buffer_capacity': 50000,
        'batch_size': 64,
        'min_buffer_size': 1000,
        'patience': 2000,
        'use_features': True,
        'model_path': 'snake_dqn.pth',
        'load_model': False,
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=CONFIG['episodes'])
    parser.add_argument("--dim", type=int, default=CONFIG['dim'])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    
    CONFIG['episodes'] = args.episodes
    CONFIG['dim'] = args.dim
    CONFIG['load_model'] = args.load or args.eval_only
    CONFIG['use_features'] = not args.no_features
    
    device = resolve_device(args.device)
    
    print("="*60)
    print("Snake RL Training with DQN")
    print("="*60)
    print(f"Board size: {CONFIG['dim']}x{CONFIG['dim']}")
    print(f"Feature extraction: {CONFIG['use_features']}")
    print(f"Device: {device}")
    
    env = SnakeEnv(
        startingBlocks=CONFIG['starting_blocks'],
        dim=CONFIG['dim'],
        distToBorder=3,
    )
    
    obs, _ = env.reset()
    state_size = obs.flatten().shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    agent = DQNAgent(
        state_size,
        action_size,
        learning_rate=CONFIG['learning_rate'],
        gamma=CONFIG['gamma'],
        epsilon_start=CONFIG['epsilon_start'],
        epsilon_end=CONFIG['epsilon_end'],
        epsilon_decay_episodes=CONFIG['epsilon_decay_episodes'],
        use_target_network=True,
        target_update_freq=CONFIG['target_update_freq'],
        buffer_capacity=CONFIG['buffer_capacity'],
        device=device,
        use_amp=not args.no_amp,
    )
    
    if CONFIG['load_model']:
        agent.load(CONFIG['model_path'])
    
    try:
        if not args.eval_only:
            rewards, lengths, losses = train_agent(
                env, agent,
                episodes=CONFIG['episodes'],
                max_steps=CONFIG['max_steps'],
                save_every=CONFIG['save_every'],
                model_path=CONFIG['model_path'],
                render_every=CONFIG['render_every'],
                min_buffer=CONFIG['min_buffer_size'],
                batch_size=CONFIG['batch_size'],
                patience=CONFIG['patience']
            )
            
            plot_training_results(rewards, lengths, losses)
            
            evaluate_agent(env, agent, num_episodes=CONFIG['eval_episodes'], render=False)
        else:
            if not agent.load(CONFIG['model_path']):
                print("Cannot evaluate without a trained model!")
            else:
                evaluate_agent(env, agent, num_episodes=CONFIG['eval_episodes'], render=args.render)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted")
        agent.save(CONFIG['model_path'])
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        agent.save(CONFIG['model_path'].replace('.pth', '_error.pth'))
    finally:
        env.close()
        print("\nTraining complete!")