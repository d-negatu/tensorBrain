#!/usr/bin/env python3
"""
Reinforcement Learning for TensorBrain
Deep Q-Network (DQN) and RL environments
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
import time
from collections import deque

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU, SGD, mse_loss
from optimizers import Adam


class DQN(Module):
    """Deep Q-Network for reinforcement learning"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Q-network
        self.q_network = Sequential(
            Linear(state_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, action_size)
        )
    
    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through Q-network"""
        return self.q_network(state)
    
    def get_action(self, state: Tensor, epsilon: float = 0.1) -> int:
        """Get action using epsilon-greedy policy"""
        if random.random() < epsilon:
            # Random action
            return random.randint(0, self.action_size - 1)
        else:
            # Greedy action
            q_values = self.forward(state)
            return np.argmax(q_values.data)
    
    def get_q_values(self, state: Tensor) -> Tensor:
        """Get Q-values for state"""
        return self.forward(state)


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class SimpleEnvironment:
    """Simple environment for testing RL"""
    
    def __init__(self, size: int = 5):
        self.size = size
        self.state = np.zeros(size)
        self.goal = size - 1
        self.position = 0
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.position = 0
        self.state = np.zeros(self.size)
        self.state[self.position] = 1.0
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Take action in environment"""
        # Action 0: move left, Action 1: move right
        if action == 0 and self.position > 0:
            self.position -= 1
        elif action == 1 and self.position < self.size - 1:
            self.position += 1
        
        # Update state
        self.state = np.zeros(self.size)
        self.state[self.position] = 1.0
        
        # Calculate reward
        if self.position == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.1
            done = False
        
        return self.state.copy(), reward, done
    
    def render(self):
        """Render environment"""
        env_str = "[" + " ".join(["G" if i == self.goal else "A" if i == self.position else "." 
                                 for i in range(self.size)]) + "]"
        print(env_str)


def train_dqn(env: SimpleEnvironment, dqn: DQN, num_episodes: int = 100, 
             batch_size: int = 32, gamma: float = 0.99, epsilon: float = 0.1,
             epsilon_decay: float = 0.995, min_epsilon: float = 0.01) -> Dict[str, Any]:
    """Train DQN agent"""
    print("🚀 Training DQN Agent...")
    
    # Initialize replay buffer and optimizer
    replay_buffer = ReplayBuffer(capacity=1000)
    optimizer = Adam(dqn.parameters(), lr=0.001)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:  # Max steps per episode
            # Get action
            state_tensor = Tensor(state.reshape(1, -1), requires_grad=False)
            action = dqn.get_action(state_tensor, epsilon)
            
            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            # Store experience
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train if enough experiences
            if len(replay_buffer) > batch_size:
                # Sample batch
                batch = replay_buffer.sample(batch_size)
                
                # Prepare batch data
                states = np.array([exp[0] for exp in batch])
                actions = np.array([exp[1] for exp in batch])
                rewards = np.array([exp[2] for exp in batch])
                next_states = np.array([exp[3] for exp in batch])
                dones = np.array([exp[4] for exp in batch])
                
                # Convert to tensors
                states_tensor = Tensor(states, requires_grad=False)
                next_states_tensor = Tensor(next_states, requires_grad=False)
                
                # Get current Q-values
                current_q_values = dqn.get_q_values(states_tensor)
                
                # Get next Q-values
                next_q_values = dqn.get_q_values(next_states_tensor)
                
                # Calculate target Q-values
                target_q_values = current_q_values.data.copy()
                for i in range(batch_size):
                    if dones[i]:
                        target_q_values[i, actions[i]] = rewards[i]
                    else:
                        target_q_values[i, actions[i]] = rewards[i] + gamma * np.max(next_q_values.data[i])
                
                target_tensor = Tensor(target_q_values, requires_grad=False)
                
                # Calculate loss
                loss = mse_loss(current_q_values, target_tensor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            if done:
                break
            
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode:3d}: Reward = {total_reward:6.2f}, "
                  f"Steps = {steps:2d}, Avg Reward = {avg_reward:6.2f}")
    
    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "final_epsilon": epsilon,
        "avg_reward": np.mean(episode_rewards[-10:])
    }


def benchmark_rl(env: SimpleEnvironment, dqn: DQN, num_episodes: int = 10) -> Dict[str, float]:
    """Benchmark RL performance"""
    print("📊 Benchmarking RL Performance...")
    
    # Test trained agent
    total_rewards = []
    total_steps = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:
            state_tensor = Tensor(state.reshape(1, -1), requires_grad=False)
            action = dqn.get_action(state_tensor, epsilon=0.0)  # No exploration
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
            
            state = next_state
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
    
    return {
        "avg_reward": np.mean(total_rewards),
        "avg_steps": np.mean(total_steps),
        "success_rate": np.mean([r > 0 for r in total_rewards]),
        "total_episodes": num_episodes
    }


if __name__ == "__main__":
    print("🎮 TensorBrain Reinforcement Learning (DQN)")
    print("=" * 50)
    
    # Create environment
    env = SimpleEnvironment(size=5)
    print(f"Environment created: {env.size}x1 grid")
    
    # Create DQN agent
    dqn = DQN(state_size=env.size, action_size=2, hidden_size=32)
    print(f"DQN created: {sum(param.data.size for param in dqn.parameters())} parameters")
    
    # Train agent
    training_results = train_dqn(env, dqn, num_episodes=100, batch_size=16)
    
    # Benchmark performance
    benchmark_results = benchmark_rl(env, dqn, num_episodes=10)
    
    print("\n📊 RL Benchmark Results:")
    print(f"Average reward: {benchmark_results['avg_reward']:.2f}")
    print(f"Average steps: {benchmark_results['avg_steps']:.2f}")
    print(f"Success rate: {benchmark_results['success_rate']:.2%}")
    print(f"Final epsilon: {training_results['final_epsilon']:.3f}")
    
    # Test agent
    print("\n🧪 Testing Trained Agent:")
    state = env.reset()
    env.render()
    
    for step in range(10):
        state_tensor = Tensor(state.reshape(1, -1), requires_grad=False)
        action = dqn.get_action(state_tensor, epsilon=0.0)
        next_state, reward, done = env.step(action)
        env.render()
        
        if done:
            print(f"Goal reached in {step + 1} steps!")
            break
        
        state = next_state
    
    print("\n🎉 Reinforcement Learning is working!")
    print("📝 Next steps:")
    print("   • Add more complex environments")
    print("   • Implement double DQN")
    print("   • Add prioritized experience replay")
    print("   • Implement dueling DQN")
    print("   • Add multi-agent RL")
    print("   • Implement policy gradient methods")
    print("   • Add continuous action spaces")
