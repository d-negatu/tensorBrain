#!/usr/bin/env python3
"""
AI Agent System for TensorBrain
Real-time AI agents that can reason and act
"""

import numpy as np
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import threading
import queue

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU, SGD, mse_loss
from rl import DQN, SimpleEnvironment
from real_llm import RealLLM, Tokenizer


@dataclass
class AgentState:
    """Agent state representation"""
    position: np.ndarray
    velocity: np.ndarray
    energy: float
    memory: Dict[str, Any]
    timestamp: float


@dataclass
class AgentAction:
    """Agent action"""
    action_type: str
    parameters: Dict[str, Any]
    confidence: float
    timestamp: float


@dataclass
class AgentObservation:
    """Agent observation of environment"""
    sensors: Dict[str, np.ndarray]
    objects: List[Dict[str, Any]]
    rewards: Dict[str, float]
    timestamp: float


class AgentMemory:
    """Agent memory system"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memories = []
        self.importance_weights = []
        
    def store(self, observation: AgentObservation, action: AgentAction, reward: float):
        """Store experience in memory"""
        memory = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "timestamp": time.time()
        }
        
        # Calculate importance weight based on reward
        importance = abs(reward) + 0.1  # Base importance
        
        self.memories.append(memory)
        self.importance_weights.append(importance)
        
        # Maintain capacity
        if len(self.memories) > self.capacity:
            # Remove least important memory
            min_idx = np.argmin(self.importance_weights)
            self.memories.pop(min_idx)
            self.importance_weights.pop(min_idx)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories"""
        # Simplified retrieval - return most recent memories
        return self.memories[-k:] if len(self.memories) >= k else self.memories
    
    def update_importance(self, memory_idx: int, new_importance: float):
        """Update importance weight of a memory"""
        if 0 <= memory_idx < len(self.importance_weights):
            self.importance_weights[memory_idx] = new_importance


class AgentPerception:
    """Agent perception system"""
    
    def __init__(self, sensor_range: float = 10.0):
        self.sensor_range = sensor_range
        self.sensors = {
            "vision": self._vision_sensor,
            "audio": self._audio_sensor,
            "touch": self._touch_sensor
        }
        
    def _vision_sensor(self, agent_state: AgentState, environment: Dict[str, Any]) -> np.ndarray:
        """Vision sensor - detect objects in range"""
        # Simplified vision - detect objects within range
        objects = environment.get("objects", [])
        vision_data = np.zeros(10)  # 10-dimensional vision vector
        
        for i, obj in enumerate(objects[:10]):
            if i < len(vision_data):
                # Calculate distance to object
                obj_pos = np.array(obj.get("position", [0, 0]))
                distance = np.linalg.norm(agent_state.position - obj_pos)
                
                if distance <= self.sensor_range:
                    # Object is visible
                    vision_data[i] = 1.0 - (distance / self.sensor_range)
        
        return vision_data
    
    def _audio_sensor(self, agent_state: AgentState, environment: Dict[str, Any]) -> np.ndarray:
        """Audio sensor - detect sounds"""
        # Simplified audio - detect sound sources
        sounds = environment.get("sounds", [])
        audio_data = np.zeros(5)  # 5-dimensional audio vector
        
        for i, sound in enumerate(sounds[:5]):
            if i < len(audio_data):
                sound_pos = np.array(sound.get("position", [0, 0]))
                distance = np.linalg.norm(agent_state.position - sound_pos)
                
                if distance <= self.sensor_range:
                    audio_data[i] = sound.get("intensity", 0.0) * (1.0 - distance / self.sensor_range)
        
        return audio_data
    
    def _touch_sensor(self, agent_state: AgentState, environment: Dict[str, Any]) -> np.ndarray:
        """Touch sensor - detect collisions"""
        # Simplified touch - detect collisions
        obstacles = environment.get("obstacles", [])
        touch_data = np.zeros(4)  # 4 directions: up, down, left, right
        
        for obstacle in obstacles:
            obs_pos = np.array(obstacle.get("position", [0, 0]))
            distance = np.linalg.norm(agent_state.position - obs_pos)
            
            if distance < 1.0:  # Collision threshold
                # Determine direction of collision
                direction = obs_pos - agent_state.position
                if abs(direction[0]) > abs(direction[1]):
                    touch_data[2 if direction[0] < 0 else 3] = 1.0  # Left or right
                else:
                    touch_data[0 if direction[1] > 0 else 1] = 1.0  # Up or down
        
        return touch_data
    
    def perceive(self, agent_state: AgentState, environment: Dict[str, Any]) -> AgentObservation:
        """Perceive environment using all sensors"""
        sensors_data = {}
        
        for sensor_name, sensor_func in self.sensors.items():
            sensors_data[sensor_name] = sensor_func(agent_state, environment)
        
        # Extract objects and rewards from environment
        objects = environment.get("objects", [])
        rewards = environment.get("rewards", {})
        
        observation = AgentObservation(
            sensors=sensors_data,
            objects=objects,
            rewards=rewards,
            timestamp=time.time()
        )
        
        return observation


class AgentReasoning:
    """Agent reasoning system"""
    
    def __init__(self, d_model: int = 128):
        self.d_model = d_model
        
        # Reasoning networks
        self.situation_analysis = Sequential(
            Linear(19, d_model),  # 10 vision + 5 audio + 4 touch
            ReLU(),
            Linear(d_model, d_model // 2),
            ReLU(),
            Linear(d_model // 2, 10)  # Situation categories
        )
        
        self.action_planning = Sequential(
            Linear(10 + d_model, d_model),  # Situation + memory
            ReLU(),
            Linear(d_model, d_model // 2),
            ReLU(),
            Linear(d_model // 2, 8)  # Action types
        )
        
        self.confidence_estimator = Sequential(
            Linear(8, 32),
            ReLU(),
            Linear(32, 1)
        )
        
    def analyze_situation(self, observation: AgentObservation) -> Dict[str, Any]:
        """Analyze current situation"""
        # Combine sensor data
        sensor_data = np.concatenate([
            observation.sensors["vision"],
            observation.sensors["audio"],
            observation.sensors["touch"]
        ])
        
        # Analyze situation
        sensor_tensor = Tensor(sensor_data.reshape(1, -1), requires_grad=False)
        situation_logits = self.situation_analysis(sensor_tensor)
        
        # Convert to probabilities
        situation_probs = np.exp(situation_logits.data) / np.sum(np.exp(situation_logits.data))
        situation_category = np.argmax(situation_probs)
        
        return {
            "category": situation_category,
            "probabilities": situation_probs,
            "confidence": np.max(situation_probs)
        }
    
    def plan_action(self, situation: Dict[str, Any], memory_context: np.ndarray) -> AgentAction:
        """Plan action based on situation and memory"""
        # Combine situation and memory
        situation_vec = np.array([situation["category"]])
        combined_input = np.concatenate([situation_vec, memory_context])
        
        # Plan action
        input_tensor = Tensor(combined_input.reshape(1, -1), requires_grad=False)
        action_logits = self.action_planning(input_tensor)
        
        # Estimate confidence
        confidence_tensor = Tensor(action_logits.data, requires_grad=False)
        confidence = self.confidence_estimator(confidence_tensor)
        
        # Select action
        action_type = np.argmax(action_logits.data)
        action_confidence = float(confidence.data[0])
        
        # Action parameters based on type
        action_parameters = self._get_action_parameters(action_type, situation)
        
        return AgentAction(
            action_type=f"action_{action_type}",
            parameters=action_parameters,
            confidence=action_confidence,
            timestamp=time.time()
        )
    
    def _get_action_parameters(self, action_type: int, situation: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for action type"""
        if action_type == 0:  # Move forward
            return {"direction": "forward", "speed": 1.0}
        elif action_type == 1:  # Turn left
            return {"direction": "left", "angle": 90}
        elif action_type == 2:  # Turn right
            return {"direction": "right", "angle": 90}
        elif action_type == 3:  # Stop
            return {"direction": "stop", "speed": 0.0}
        elif action_type == 4:  # Investigate
            return {"target": "nearest_object", "duration": 5.0}
        elif action_type == 5:  # Avoid
            return {"target": "obstacle", "distance": 2.0}
        elif action_type == 6:  # Collect
            return {"target": "reward", "priority": "high"}
        else:  # Wait
            return {"duration": 1.0}


class AIAgent:
    """Complete AI agent system"""
    
    def __init__(self, agent_id: str, initial_position: np.ndarray = np.array([0, 0])):
        self.agent_id = agent_id
        self.state = AgentState(
            position=initial_position,
            velocity=np.array([0, 0]),
            energy=100.0,
            memory={},
            timestamp=time.time()
        )
        
        # Agent components
        self.memory = AgentMemory(capacity=500)
        self.perception = AgentPerception(sensor_range=5.0)
        self.reasoning = AgentReasoning(d_model=64)
        
        # Action history
        self.action_history = []
        self.observation_history = []
        
        print(f"🤖 AIAgent {agent_id} initialized:")
        print(f"   Initial position: {initial_position}")
        print(f"   Memory capacity: {self.memory.capacity}")
        print(f"   Sensor range: {self.perception.sensor_range}")
    
    def perceive(self, environment: Dict[str, Any]) -> AgentObservation:
        """Perceive environment"""
        observation = self.perception.perceive(self.state, environment)
        self.observation_history.append(observation)
        return observation
    
    def reason(self, observation: AgentObservation) -> AgentAction:
        """Reason about situation and plan action"""
        # Analyze situation
        situation = self.reasoning.analyze_situation(observation)
        
        # Get memory context (simplified)
        memory_context = np.random.randn(self.reasoning.d_model)  # Simplified
        
        # Plan action
        action = self.reasoning.plan_action(situation, memory_context)
        
        return action
    
    def act(self, action: AgentAction, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action in environment"""
        self.action_history.append(action)
        
        # Update agent state based on action
        if action.action_type == "action_0":  # Move forward
            self.state.position += np.array([0, 1])
            self.state.energy -= 1.0
        elif action.action_type == "action_1":  # Turn left
            self.state.velocity = np.array([-1, 0])
        elif action.action_type == "action_2":  # Turn right
            self.state.velocity = np.array([1, 0])
        elif action.action_type == "action_3":  # Stop
            self.state.velocity = np.array([0, 0])
        
        # Calculate reward
        reward = self._calculate_reward(action, environment)
        
        # Store experience in memory
        if self.observation_history:
            last_observation = self.observation_history[-1]
            self.memory.store(last_observation, action, reward)
        
        return {
            "action": action,
            "reward": reward,
            "new_position": self.state.position.copy(),
            "energy": self.state.energy
        }
    
    def _calculate_reward(self, action: AgentAction, environment: Dict[str, Any]) -> float:
        """Calculate reward for action"""
        reward = 0.0
        
        # Check for rewards in environment
        rewards = environment.get("rewards", {})
        for reward_name, reward_value in rewards.items():
            if np.linalg.norm(self.state.position - np.array(reward_value.get("position", [0, 0]))) < 1.0:
                reward += reward_value.get("value", 0.0)
        
        # Penalty for energy consumption
        if action.action_type in ["action_0", "action_1", "action_2"]:
            reward -= 0.1
        
        # Bonus for exploration
        if action.action_type == "action_4":  # Investigate
            reward += 0.5
        
        return reward
    
    def step(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Complete agent step: perceive, reason, act"""
        # Perceive environment
        observation = self.perceive(environment)
        
        # Reason about situation
        action = self.reason(observation)
        
        # Act in environment
        result = self.act(action, environment)
        
        return {
            "observation": observation,
            "action": action,
            "result": result,
            "agent_state": self.state
        }


class MultiAgentSystem:
    """Multi-agent system with coordination"""
    
    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.agents = []
        self.environment = self._create_environment()
        
        # Create agents
        for i in range(num_agents):
            initial_pos = np.random.randn(2) * 5  # Random initial position
            agent = AIAgent(f"agent_{i}", initial_pos)
            self.agents.append(agent)
        
        print(f"🌐 MultiAgentSystem initialized:")
        print(f"   Number of agents: {num_agents}")
        print(f"   Environment size: 20x20")
    
    def _create_environment(self) -> Dict[str, Any]:
        """Create simulation environment"""
        return {
            "objects": [
                {"position": [5, 5], "type": "food", "value": 10},
                {"position": [-3, 2], "type": "water", "value": 5},
                {"position": [0, -4], "type": "shelter", "value": 15}
            ],
            "obstacles": [
                {"position": [2, 2], "type": "wall"},
                {"position": [-1, -1], "type": "rock"}
            ],
            "sounds": [
                {"position": [3, 3], "intensity": 0.8, "type": "alarm"}
            ],
            "rewards": {
                "food": {"position": [5, 5], "value": 10.0},
                "water": {"position": [-3, 2], "value": 5.0},
                "shelter": {"position": [0, -4], "value": 15.0}
            }
        }
    
    def run_simulation(self, num_steps: int = 100) -> Dict[str, Any]:
        """Run multi-agent simulation"""
        print(f"🚀 Running multi-agent simulation for {num_steps} steps...")
        
        simulation_results = []
        
        for step in range(num_steps):
            step_results = []
            
            # Each agent takes a step
            for agent in self.agents:
                result = agent.step(self.environment)
                step_results.append(result)
            
            simulation_results.append(step_results)
            
            # Print progress
            if step % 20 == 0:
                print(f"Step {step}: Agents at positions {[agent.state.position for agent in self.agents]}")
        
        # Calculate final statistics
        final_stats = self._calculate_final_stats()
        
        return {
            "simulation_results": simulation_results,
            "final_stats": final_stats,
            "num_steps": num_steps,
            "num_agents": self.num_agents
        }
    
    def _calculate_final_stats(self) -> Dict[str, Any]:
        """Calculate final simulation statistics"""
        total_rewards = []
        final_positions = []
        energy_levels = []
        
        for agent in self.agents:
            # Calculate total reward from memory
            total_reward = sum(memory["reward"] for memory in agent.memory.memories)
            total_rewards.append(total_reward)
            
            final_positions.append(agent.state.position.copy())
            energy_levels.append(agent.state.energy)
        
        return {
            "total_rewards": total_rewards,
            "avg_reward": np.mean(total_rewards),
            "final_positions": final_positions,
            "energy_levels": energy_levels,
            "survival_rate": sum(1 for energy in energy_levels if energy > 0) / len(energy_levels)
        }


def demo_ai_agent():
    """Demonstrate AI agent system"""
    print("🤖 TensorBrain AI Agent System Demo")
    print("=" * 50)
    
    # Create single agent
    agent = AIAgent("test_agent", initial_position=np.array([0, 0]))
    
    # Create environment
    environment = {
        "objects": [
            {"position": [3, 3], "type": "food", "value": 10},
            {"position": [-2, 1], "type": "water", "value": 5}
        ],
        "obstacles": [
            {"position": [1, 1], "type": "wall"}
        ],
        "sounds": [
            {"position": [2, 2], "intensity": 0.6, "type": "signal"}
        ],
        "rewards": {
            "food": {"position": [3, 3], "value": 10.0},
            "water": {"position": [-2, 1], "value": 5.0}
        }
    }
    
    # Test single agent
    print("\n🧪 Testing single agent...")
    for step in range(10):
        result = agent.step(environment)
        print(f"Step {step}: Action = {result['action'].action_type}, "
              f"Position = {result['result']['new_position']}, "
              f"Reward = {result['result']['reward']:.2f}")
    
    # Test multi-agent system
    print("\n🌐 Testing multi-agent system...")
    multi_agent_system = MultiAgentSystem(num_agents=3)
    simulation_results = multi_agent_system.run_simulation(num_steps=50)
    
    print(f"\n📊 Multi-Agent Simulation Results:")
    print(f"Average reward: {simulation_results['final_stats']['avg_reward']:.2f}")
    print(f"Survival rate: {simulation_results['final_stats']['survival_rate']:.2%}")
    print(f"Final positions: {simulation_results['final_stats']['final_positions']}")
    
    print("\n🎉 AI Agent System is working!")
    print("📝 Next steps:")
    print("   • Add communication between agents")
    print("   • Implement swarm intelligence")
    print("   • Add learning from other agents")
    print("   • Implement hierarchical agents")
    print("   • Add natural language interaction")
    print("   • Implement goal-oriented behavior")
    
    return simulation_results


if __name__ == "__main__":
    demo_ai_agent()
