# Snake Reinforcement Learning

A Snake game implementation trained using Deep Q-Network (DQN) reinforcement learning. The agent learns to play Snake by maximizing rewards through deep neural networks.

## 🎯 What It Does

This project trains an AI agent to play the classic Snake game using reinforcement learning. The agent learns optimal strategies through:
- **DQN (Deep Q-Network)**: Neural network-based Q-learning
- **Experience Replay**: Efficient learning from past experiences
- **Target Network**: Stable training with periodic updates
- **Reward Shaping**: Incentives for approaching food, surviving, and growing

The trained agent navigates a 10x10 grid, avoiding walls and its own body while collecting power-ups to grow longer.

## 📊 Current Best Scores

**Last 10 episodes:**
- **Average Reward:** 76.39 ± 361.09
- **Average Steps:** 878.4 ± 294.2
- **Best Reward:** 263.96

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (project uses `snake_rl` environment)
- Node.js (for web interface)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ninja-boldo/snake.git
cd snake
```

2. **Set up Python environment:**
```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

3. **Install Node.js dependencies (for visualization):**
```bash
npm install
```

### Training the Agent

**Start training from scratch:**
```bash
python rl/train.py
```

**Resume training from saved model:**
```bash
python rl/train.py --load
```

**Training options:**
```bash
python rl/train.py --episodes 20000 --dim 10 --device auto
```

Available options:
- `--episodes N`: Number of training episodes (default: 20000)
- `--dim N`: Board size (NxN grid, default: 10)
- `--device`: Device to use (`auto`, `cpu`, `cuda`, `mps`)
- `--no-amp`: Disable automatic mixed precision (for CUDA)
- `--load`: Load existing model before training

### Evaluating the Agent

**Evaluate trained model:**
```bash
python rl/train.py --eval-only
```

**Evaluate with visualization:**
```bash
python rl/train.py --eval-only --render
```

### Running the Web Interface

```bash
npm run dev
```

Then open your browser to `http://localhost:5173`

## 📁 Project Structure

```
snake/
├── rl/
│   ├── train.py           # Main DQN training script
│   ├── env_local.py       # Gymnasium environment wrapper
│   ├── env_js.py          # JavaScript integration environment
│   └── train_js.py        # Training with JS visualization
├── game_py/
│   ├── snake_engine.py    # Core Snake game logic
│   └── tests.py           # Game engine tests
├── src/
│   └── main.ts            # Web interface (TypeScript)
├── public/                # Static web assets
├── requirements.txt       # Python dependencies
├── package.json           # Node.js dependencies
├── snake_dqn.pth         # Latest trained model
├── snake_dqn_best.pth    # Best performing model
└── training_results.png  # Training visualization
```

## 🧠 Technical Details

### Neural Network Architecture
- **Input:** 19 features (direction encoding, danger detection, food location, snake position)
- **Hidden Layers:** 256 → 256 → 128 neurons with ReLU activation
- **Output:** 4 Q-values (one for each direction: up, right, down, left)

### Training Configuration
- **Learning Rate:** 0.0003
- **Discount Factor (γ):** 0.95
- **Epsilon Decay:** 1.0 → 0.01 over 5000 episodes
- **Replay Buffer:** 50,000 experiences
- **Batch Size:** 64
- **Target Network Update:** Every 500 steps

### Reward System
- **Power-up collection:** +10.0
- **Moving closer to food:** +0.1
- **Moving away from food:** -0.1
- **Survival:** +0.01 per step
- **Collision (death):** -10.0
- **Near danger:** -0.05

### State Representation (Features)
1. Current direction (one-hot encoded)
2. Danger detection in 4 directions
3. Food direction relative to head
4. Normalized food position
5. Snake length (normalized)
6. Head position (normalized)
7. Tail position (normalized)

## 🎮 Usage Examples

### Train with custom parameters
```bash
python rl/train.py --episodes 10000 --dim 15 --device cuda
```

### Evaluate multiple episodes
```bash
python rl/train.py --eval-only --render
```

### Use specific features
```bash
python rl/train.py --no-features  # Use raw grid instead of features
```

## 📈 Monitoring Training

During training, the script outputs:
- Episode number and progress
- Average reward over last 100 episodes
- Average episode length
- Training loss
- Current epsilon (exploration rate)

Training plots are automatically saved to `training_results.png` showing:
- Reward progression
- Episode lengths
- Training loss
- Reward distribution

## 🔧 Dependencies

### Python
- PyTorch (deep learning framework)
- Gymnasium (RL environment interface)
- NumPy (numerical computing)
- Matplotlib (visualization)
- Pandas (data analysis)

### JavaScript
- Vite (build tool and dev server)
- TypeScript (type-safe JavaScript)
- WebSocket (real-time communication)

## 🐛 Troubleshooting

**CUDA out of memory:**
```bash
python rl/train.py --device cpu --no-amp
```

**Model not found:**
Ensure you've trained a model first or specify the correct path to an existing model.

**Web interface not loading:**
Make sure to run `npm install` and check that port 5173 is available.

## 📝 Notes

- Models are automatically saved every 500 episodes
- Best performing model is saved as `snake_dqn_best.pth`
- Training can be interrupted with Ctrl+C and will save progress
- The agent uses epsilon-greedy exploration during training
- Evaluation mode disables exploration for deterministic behavior

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

ISC
