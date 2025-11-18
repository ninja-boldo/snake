import argparse
import time
import numpy as np
import torch
from env_js import SnakeEnv
from train_js import DQNAgent, resolve_device


def run_inference(
    env,
    agent,
    num_episodes=None,
    max_steps=1000,
    verbose=True
):
    """
    Run inference with a trained agent.
    
    Args:
        env: The Snake environment
        agent: Trained DQN agent
        num_episodes: Number of episodes to run (None = infinite)
        max_steps: Maximum steps per episode
        verbose: Print episode statistics
    """
    print("\n" + "="*60)
    print("ðŸŽ® INFERENCE MODE - Running Trained Agent")
    print("="*60)
    print(f"Episodes: {'Infinite (Ctrl+C to stop)' if num_episodes is None else num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Model epsilon: {agent.epsilon:.4f}")
    print("="*60 + "\n")
    
    episode = 0
    total_rewards = []
    total_lengths = []
    
    try:
        while num_episodes is None or episode < num_episodes:
            episode += 1
            
            # Reset environment
            obs, info = env.reset()
            state = obs.astype(np.float32).flatten()
            total_reward = 0
            
            # Run episode
            for step in range(max_steps):
                # Get action from agent (no exploration)
                action = agent.act(state, inference_mode=True)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = obs.astype(np.float32).flatten()
                total_reward += reward
                state = next_state
                
                # Check if episode ended
                if terminated or truncated:
                    break
            
            # Record statistics
            total_rewards.append(total_reward)
            total_lengths.append(step + 1)
            
            # Print episode results
            if verbose:
                print(f"Episode {episode:4d} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Steps: {step + 1:4d}")
            
            # Print rolling statistics every 10 episodes
            if episode > 0 and episode % 10 == 0 and verbose:
                recent_rewards = total_rewards[-10:]
                recent_lengths = total_lengths[-10:]
                print(f"\nðŸ“Š Last 10 episodes:")
                print(f"   Avg Reward: {np.mean(recent_rewards):7.2f} Â± {np.std(recent_rewards):.2f}")
                print(f"   Avg Steps:  {np.mean(recent_lengths):7.1f} Â± {np.std(recent_lengths):.1f}")
                print(f"   Best Reward: {max(recent_rewards):7.2f}\n")
    
    except KeyboardInterrupt:
        print("\n\nâ¹ï¸  Inference stopped by user")
    
    # Final statistics
    if len(total_rewards) > 0:
        print("\n" + "="*60)
        print("ðŸ“ˆ FINAL STATISTICS")
        print("="*60)
        print(f"Total Episodes: {len(total_rewards)}")
        print(f"Average Reward: {np.mean(total_rewards):.2f} Â± {np.std(total_rewards):.2f}")
        print(f"Average Steps:  {np.mean(total_lengths):.1f} Â± {np.std(total_lengths):.1f}")
        print(f"Best Reward:    {max(total_rewards):.2f}")
        print(f"Worst Reward:   {min(total_rewards):.2f}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained Snake RL agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="snake_dqn_best.pth",
        help="Path to trained model file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to run (default: infinite)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=20,
        help="Board dimension (must match training)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default="ws://localhost:3030",
        help="WebSocket server URL"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress episode-by-episode output"
    )
    
    args = parser.parse_args()
    
    # Resolve device
    device = resolve_device(args.device)
    
    print("\n" + "="*60)
    print("ðŸ Snake RL Inference")
    print("="*60)
    print(f"Model:      {args.model}")
    print(f"Board size: {args.dim}x{args.dim}")
    print(f"Device:     {device}")
    print(f"WebSocket:  {args.ws_url}")
    print("="*60)
    
    # Create environment
    try:
        env = SnakeEnv(
            startingBlocks=3,
            dim=args.dim,
            distToBorder=3,
            ws_url=args.ws_url
        )
    except Exception as e:
        print(f"\nâŒ Failed to connect to game: {e}")
        print("\nMake sure:")
        print("  1. WebSocket server is running (node websocket-server.js)")
        print("  2. Game is open in browser")
        return
    
    # Get state and action dimensions
    obs, _ = env.reset()
    state_size = obs.flatten().shape[0]
    action_size = env.action_space.n
    
    print(f"\nState size:  {state_size}")
    print(f"Action size: {action_size}")
    
    # Create agent
    agent = DQNAgent(
        state_size,
        action_size,
        device=device,
        use_amp=False,  # Not needed for inference
    )
    
    # Load trained model
    print(f"\nðŸ“‚ Loading model from {args.model}...")
    if not agent.load(args.model):
        print(f"\nâŒ Failed to load model from {args.model}")
        print("Train a model first using: python train_js.py")
        env.close()
        return
    
    print("âœ… Model loaded successfully!")
    
    # Run inference
    try:
        run_inference(
            env,
            agent,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            verbose=not args.quiet
        )
    finally:
        env.close()
        print("ðŸ§¹ Environment closed")


if __name__ == "__main__":
    main()