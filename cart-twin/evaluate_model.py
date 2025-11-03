#!/usr/bin/env python3
"""
Evaluation script for trained CAR T-cell PPO models
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from car_t_env import CarTManufactureEnv

def load_and_evaluate(model_path, platform="grex", num_episodes=10, seed=123):
    """Load trained model and evaluate it"""
    
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    env = CarTManufactureEnv(platform=platform, episode_days=12, seed=seed)
    
    results = []
    
    print(f"Evaluating {platform} model on {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        # Track detailed metrics
        metrics = {
            'time': [],
            'viable_cells': [],
            'exhausted_cells': [],
            'glucose': [],
            'lactate': [],
            'il2': [],
            'bead_occupancy': [],
            'exhaustion_fraction': [],
            'rewards': [],
            'actions': [],
            'resource_usage': {
                'total_il2_used': 0,
                'total_media_used': 0,
                'total_beads_used': 0
            }
        }
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record metrics
            day = step_count * env.p["dt_h"] / 24.0
            metrics['time'].append(day)
            metrics['viable_cells'].append(obs[0])
            metrics['exhausted_cells'].append(obs[1])
            metrics['glucose'].append(obs[2])
            metrics['lactate'].append(obs[3])
            metrics['il2'].append(obs[4])
            metrics['bead_occupancy'].append(obs[5])
            metrics['exhaustion_fraction'].append(info['exhaustion_fraction'])
            metrics['rewards'].append(reward)
            metrics['actions'].append(action.copy())
            
            # Track resource usage
            if action[1] > 0:  # IL-2 used
                il2_amounts = [0, 50, 150, 300]
                metrics['resource_usage']['total_il2_used'] += il2_amounts[action[1]]
            if action[2] > 0:  # Media used
                media_fractions = [0, 0.25, 0.50, 1.0]
                metrics['resource_usage']['total_media_used'] += media_fractions[action[2]]
            if action[0] in [1, 2]:  # Beads used
                metrics['resource_usage']['total_beads_used'] += 1
            
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

def evaluate_targets(results, platform):
    """Evaluate results against biological targets"""
    
    print(f"\n{'='*60}")
    print(f"TARGET EVALUATION - {platform.upper()} PLATFORM")
    print(f"{'='*60}")
    
    # Define targets based on platform
    if platform == "grex":
        target_expansion = 50  # 50x expansion over 10-12 days
        target_exhaustion = 0.15  # <15% exhaustion
        target_glucose_min = 5.0  # >5 mM glucose
        target_lactate_max = 20.0  # <20 mM lactate
    else:  # stirred_tank
        target_expansion = 100  # 100x expansion over 10-12 days
        target_exhaustion = 0.20  # <20% exhaustion
        target_glucose_min = 5.0  # >5 mM glucose
        target_lactate_max = 20.0  # <20 mM lactate
    
    # Calculate metrics
    viable_cells = [r['final_viable'] for r in results]
    exhaustion = [r['final_exhaustion'] for r in results]
    expansion_ratios = [nv / 2.0 for nv in viable_cells]  # Starting with 2M cells
    
    # Process sanity checks
    glucose_violations = []
    lactate_violations = []
    
    for result in results:
        metrics = result['metrics']
        glucose_below_threshold = sum(1 for g in metrics['glucose'] if g < target_glucose_min)
        lactate_above_threshold = sum(1 for l in metrics['lactate'] if l > target_lactate_max)
        glucose_violations.append(glucose_below_threshold)
        lactate_violations.append(lactate_above_threshold)
    
    # Resource usage analysis
    total_il2 = [sum(r['metrics']['resource_usage']['total_il2_used'] for r in results)]
    total_media = [sum(r['metrics']['resource_usage']['total_media_used'] for r in results)]
    total_beads = [sum(r['metrics']['resource_usage']['total_beads_used'] for r in results)]
    
    # Print target evaluation
    print(f"YIELD TARGETS:")
    print(f"  Target expansion: {target_expansion}x")
    print(f"  Achieved expansion: {np.mean(expansion_ratios):.1f}x (range: {np.min(expansion_ratios):.1f}-{np.max(expansion_ratios):.1f})")
    print(f"  Target met: {np.mean(expansion_ratios) >= target_expansion * 0.8:.1f} (80% of target)")
    
    print(f"\nPHENOTYPE TARGETS:")
    print(f"  Target exhaustion: <{target_exhaustion*100:.0f}%")
    print(f"  Achieved exhaustion: {np.mean(exhaustion)*100:.1f}% ± {np.std(exhaustion)*100:.1f}%")
    print(f"  Target met: {np.mean(exhaustion) <= target_exhaustion:.1f}")
    
    print(f"\nPROCESS SANITY:")
    print(f"  Glucose <{target_glucose_min} mM: {np.mean(glucose_violations):.1f} steps/episode")
    print(f"  Lactate >{target_lactate_max} mM: {np.mean(lactate_violations):.1f} steps/episode")
    print(f"  Process stable: {np.mean(glucose_violations) < 5 and np.mean(lactate_violations) < 5:.1f}")
    
    print(f"\nRESOURCE EFFICIENCY:")
    print(f"  Total IL-2 used: {np.mean(total_il2):.0f} units")
    print(f"  Total media used: {np.mean(total_media):.2f} fraction")
    print(f"  Total bead uses: {np.mean(total_beads):.1f} times")
    
    # Overall score
    yield_score = min(1.0, np.mean(expansion_ratios) / target_expansion)
    phenotype_score = max(0.0, 1.0 - np.mean(exhaustion) / target_exhaustion)
    process_score = max(0.0, 1.0 - (np.mean(glucose_violations) + np.mean(lactate_violations)) / 20)
    
    overall_score = (yield_score + phenotype_score + process_score) / 3
    
    print(f"\nOVERALL PERFORMANCE SCORE: {overall_score:.2f}/1.0")
    print(f"  Yield: {yield_score:.2f}")
    print(f"  Phenotype: {phenotype_score:.2f}")
    print(f"  Process: {process_score:.2f}")
    
    return {
        'yield_score': yield_score,
        'phenotype_score': phenotype_score,
        'process_score': process_score,
        'overall_score': overall_score,
        'expansion_ratio': np.mean(expansion_ratios),
        'exhaustion_fraction': np.mean(exhaustion),
        'glucose_violations': np.mean(glucose_violations),
        'lactate_violations': np.mean(lactate_violations)
    }

def plot_detailed_results(results, platform, episode_idx=0):
    """Plot detailed results for a specific episode"""
    
    metrics = results[episode_idx]['metrics']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Detailed Episode {episode_idx+1} Results - {platform.upper()} Platform', fontsize=16)
    
    # Cell populations
    axes[0, 0].plot(metrics['time'], metrics['viable_cells'], 'g-', label='Viable', linewidth=2)
    axes[0, 0].plot(metrics['time'], metrics['exhausted_cells'], 'r-', label='Exhausted', linewidth=2)
    axes[0, 0].set_title('Cell Populations')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Cells (×10⁶)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Nutrients
    axes[0, 1].plot(metrics['time'], metrics['glucose'], 'b-', label='Glucose', linewidth=2)
    axes[0, 1].plot(metrics['time'], metrics['lactate'], 'orange', label='Lactate', linewidth=2)
    axes[0, 1].set_title('Nutrient Levels')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Concentration (mM)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IL-2 and Bead occupancy
    ax_twin = axes[0, 2].twinx()
    axes[0, 2].plot(metrics['time'], metrics['il2'], 'purple', label='IL-2', linewidth=2)
    ax_twin.plot(metrics['time'], metrics['bead_occupancy'], 'brown', label='Bead Occupancy', linewidth=2)
    axes[0, 2].set_title('IL-2 and Bead Occupancy')
    axes[0, 2].set_xlabel('Time (days)')
    axes[0, 2].set_ylabel('IL-2 (units)', color='purple')
    ax_twin.set_ylabel('Bead Occupancy', color='brown')
    axes[0, 2].grid(True)
    
    # Exhaustion fraction
    axes[1, 0].plot(metrics['time'], [x*100 for x in metrics['exhaustion_fraction']], 'r-', linewidth=2)
    axes[1, 0].set_title('Exhaustion Fraction')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Exhaustion (%)')
    axes[1, 0].grid(True)
    
    # Actions over time
    actions = np.array(metrics['actions'])
    axes[1, 1].plot(metrics['time'], actions[:, 0], 'o-', label='Beads', markersize=4)
    axes[1, 1].plot(metrics['time'], actions[:, 1], 's-', label='IL-2', markersize=4)
    axes[1, 1].plot(metrics['time'], actions[:, 2], '^-', label='Media', markersize=4)
    axes[1, 1].set_title('Actions Over Time')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Action Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Cumulative reward
    cumulative_rewards = np.cumsum(metrics['rewards'])
    axes[1, 2].plot(metrics['time'], cumulative_rewards, 'g-', linewidth=2)
    axes[1, 2].set_title('Cumulative Reward')
    axes[1, 2].set_xlabel('Time (days)')
    axes[1, 2].set_ylabel('Cumulative Reward')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{platform}_detailed_episode_{episode_idx+1}.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_action_patterns(results, platform):
    """Analyze action patterns across episodes"""
    
    all_actions = []
    for result in results:
        all_actions.extend(result['metrics']['actions'])
    
    all_actions = np.array(all_actions)
    
    print(f"\n{platform.upper()} Action Analysis:")
    print(f"Bead actions: {np.bincount(all_actions[:, 0], minlength=4)}")
    print(f"IL-2 actions: {np.bincount(all_actions[:, 1], minlength=4)}")
    print(f"Media actions: {np.bincount(all_actions[:, 2], minlength=4)}")
    print(f"Agitation actions: {np.bincount(all_actions[:, 3], minlength=3)}")
    
    # Plot action distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Action Distribution Analysis - {platform.upper()} Platform', fontsize=16)
    
    action_names = ['Beads', 'IL-2', 'Media', 'Agitation']
    action_ranges = [4, 4, 4, 3]
    
    for i, (name, max_val) in enumerate(zip(action_names, action_ranges)):
        row, col = i // 2, i % 2
        counts = np.bincount(all_actions[:, i], minlength=max_val)
        axes[row, col].bar(range(max_val), counts, alpha=0.7)
        axes[row, col].set_title(f'{name} Actions')
        axes[row, col].set_xlabel('Action Value')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{platform}_action_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <model_path> [platform] [num_episodes]")
        print("Example: python evaluate_model.py ppo_cart_grex grex 10")
        sys.exit(1)
    
    model_path = sys.argv[1]
    platform = sys.argv[2] if len(sys.argv) > 2 else "grex"
    num_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    # Evaluate model
    results = load_and_evaluate(model_path, platform, num_episodes)
    
    # Print summary
    rewards = [r['total_reward'] for r in results]
    viable_cells = [r['final_viable'] for r in results]
    exhaustion = [r['final_exhaustion'] for r in results]
    
    print(f"\n{platform.upper()} Evaluation Summary:")
    print(f"Average Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Average Viable Cells: {np.mean(viable_cells):.1f} ± {np.std(viable_cells):.1f}")
    print(f"Average Exhaustion: {np.mean(exhaustion)*100:.1f}% ± {np.std(exhaustion)*100:.1f}%")
    
    # Evaluate against biological targets
    target_scores = evaluate_targets(results, platform)
    
    # Plot detailed results for first episode
    plot_detailed_results(results, platform, episode_idx=0)
    
    # Analyze action patterns
    analyze_action_patterns(results, platform)
