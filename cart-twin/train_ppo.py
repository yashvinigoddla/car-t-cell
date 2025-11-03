#!/usr/bin/env python3
"""
PPO training script for CAR T-cell manufacturing optimization using Stable-Baselines3
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os
from car_t_env import CarTManufactureEnv

def make_env(platform, episode_days=12, seed=None):
    """Create environment factory for vectorized training"""
    def _init():
        env = CarTManufactureEnv(platform=platform, episode_days=episode_days, seed=seed)
        return Monitor(env)
    return _init

def train_ppo_agent(platform="grex", total_timesteps=1_000_000, n_envs=8, save_path=None):
    """Train PPO agent on CAR T-cell environment"""
    
    print(f"Training PPO on {platform} platform...")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of parallel environments: {n_envs}")
    
    # Create vectorized environment
    env = make_vec_env(make_env(platform), n_envs=n_envs)
    
    # Create PPO model with tuned hyperparameters
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        n_steps=1024,           # Steps per update
        batch_size=2048,        # Batch size for training
        learning_rate=3e-4,     # Learning rate
        gamma=0.995,           # Discount factor
        gae_lambda=0.95,       # GAE lambda
        n_epochs=5,            # Number of epochs per update
        clip_range=0.2,        # PPO clip range
        ent_coef=0.01,         # Entropy coefficient
        vf_coef=0.5,           # Value function coefficient
        max_grad_norm=0.5,     # Max gradient norm
        tensorboard_log=f"./tensorboard_logs/{platform}/"
    )
    
    # Create save directory
    if save_path is None:
        save_path = f"ppo_cart_{platform}"
    
    # Create evaluation callback
    eval_env = CarTManufactureEnv(platform=platform, episode_days=12, seed=123)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=f"./models/{platform}/",
        log_path=f"./logs/{platform}/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model

def evaluate_model(model, platform="grex", num_episodes=10, seed=123):
    """Evaluate trained model"""
    
    env = CarTManufactureEnv(platform=platform, episode_days=12, seed=seed)
    
    results = []
    
    print(f"\nEvaluating {platform} model on {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        # Track metrics over time
        metrics = {
            'viable_cells': [],
            'exhausted_cells': [],
            'glucose': [],
            'lactate': [],
            'il2': [],
            'bead_occupancy': [],
            'exhaustion_fraction': [],
            'rewards': []
        }
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record metrics
            metrics['viable_cells'].append(obs[0])
            metrics['exhausted_cells'].append(obs[1])
            metrics['glucose'].append(obs[2])
            metrics['lactate'].append(obs[3])
            metrics['il2'].append(obs[4])
            metrics['bead_occupancy'].append(obs[5])
            metrics['exhaustion_fraction'].append(info['exhaustion_fraction'])
            metrics['rewards'].append(reward)
            
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        results.append({
            'episode': episode,
            'total_reward': total_reward,
            'final_viable': obs[0],
            'final_exhausted': obs[1],
            'final_exhaustion': info['exhaustion_fraction'],
            'steps': step_count,
            'metrics': metrics
        })
        
        print(f"Episode {episode+1}: Reward={total_reward:.3f}, "
              f"Viable={obs[0]:.1f}, Exhaustion={info['exhaustion_fraction']*100:.1f}%")
    
    return results

def plot_evaluation_results(results, platform):
    """Plot evaluation results"""
    
    # Extract metrics
    rewards = [r['total_reward'] for r in results]
    viable_cells = [r['final_viable'] for r in results]
    exhaustion = [r['final_exhaustion'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'PPO Evaluation Results - {platform.upper()} Platform', fontsize=16)
    
    # Plot rewards
    axes[0].bar(range(len(rewards)), rewards, alpha=0.7)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)
    
    # Plot viable cells
    axes[1].bar(range(len(viable_cells)), viable_cells, alpha=0.7, color='green')
    axes[1].set_title('Final Viable Cells')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Viable Cells (×10⁶)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot exhaustion
    exhaustion_pct = [x*100 for x in exhaustion]
    axes[2].bar(range(len(exhaustion_pct)), exhaustion_pct, alpha=0.7, color='red')
    axes[2].set_title('Final Exhaustion Fraction')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Exhaustion (%)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{platform}_ppo_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n{platform.upper()} Evaluation Summary:")
    print(f"Average Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Average Viable Cells: {np.mean(viable_cells):.1f} ± {np.std(viable_cells):.1f}")
    print(f"Average Exhaustion: {np.mean(exhaustion)*100:.1f}% ± {np.std(exhaustion)*100:.1f}%")

def compare_platforms():
    """Train and compare both platforms"""
    
    platforms = ["grex", "stirred_tank"]
    models = {}
    results = {}
    
    for platform in platforms:
        print(f"\n{'='*60}")
        print(f"Training PPO on {platform.upper()} platform")
        print(f"{'='*60}")
        
        # Train model
        model = train_ppo_agent(platform=platform, total_timesteps=500_000, n_envs=4)
        models[platform] = model
        
        # Evaluate model
        eval_results = evaluate_model(model, platform=platform, num_episodes=5)
        results[platform] = eval_results
        
        # Plot results
        plot_evaluation_results(eval_results, platform)
    
    # Compare platforms
    print(f"\n{'='*60}")
    print("PLATFORM COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'G-Rex':<15} {'Stirred Tank':<15} {'Difference':<15}")
    print("-" * 70)
    
    grex_rewards = [r['total_reward'] for r in results['grex']]
    grex_viable = [r['final_viable'] for r in results['grex']]
    grex_exhaustion = [r['final_exhaustion'] for r in results['grex']]
    
    tank_rewards = [r['total_reward'] for r in results['stirred_tank']]
    tank_viable = [r['final_viable'] for r in results['stirred_tank']]
    tank_exhaustion = [r['final_exhaustion'] for r in results['stirred_tank']]
    
    print(f"{'Avg Reward':<25} {np.mean(grex_rewards):<15.3f} {np.mean(tank_rewards):<15.3f} {np.mean(tank_rewards)-np.mean(grex_rewards):<15.3f}")
    print(f"{'Avg Viable Cells':<25} {np.mean(grex_viable):<15.1f} {np.mean(tank_viable):<15.1f} {np.mean(tank_viable)-np.mean(grex_viable):<15.1f}")
    print(f"{'Avg Exhaustion %':<25} {np.mean(grex_exhaustion)*100:<15.1f} {np.mean(tank_exhaustion)*100:<15.1f} {(np.mean(tank_exhaustion)-np.mean(grex_exhaustion))*100:<15.1f}")

if __name__ == "__main__":
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    # Run comparison
    compare_platforms()
