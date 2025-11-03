#!/usr/bin/env python3
"""
Calibration guide for CAR T-cell digital twin environment
Helps fit model parameters to real experimental data
"""

import numpy as np
import matplotlib.pyplot as plt
from car_t_env import CarTManufactureEnv, PLATFORMS

def run_calibration_simulation(platform="grex", days=12, seed=42):
    """Run simulation with current parameters to see baseline behavior"""
    
    env = CarTManufactureEnv(platform=platform, episode_days=days, seed=seed)
    obs, _ = env.reset()
    
    # Track metrics over time
    metrics = {
        'time': [],
        'viable_cells': [],
        'exhausted_cells': [],
        'glucose': [],
        'lactate': [],
        'il2': [],
        'exhaustion_fraction': []
    }
    
    # Simple heuristic strategy for calibration
    for step in range(env.steps):
        day = step * env.p["dt_h"] / 24.0
        Nv, Ne, Glu, Lac, IL2, B, age, p0, p1 = obs
        
        # Calibration strategy: minimal intervention
        if day < 1.0:
            beads = 1  # Add beads early
        else:
            beads = 0
        
        if day < 3.0:
            il2 = 1  # Low IL-2 early
        else:
            il2 = 0
        
        if Glu < 8 or Lac > 15:
            media = 2  # 50% refresh when needed
        else:
            media = 0
        
        agitation = 1  # Medium agitation
        
        action = np.array([beads, il2, media, agitation])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record metrics
        metrics['time'].append(day)
        metrics['viable_cells'].append(obs[0])
        metrics['exhausted_cells'].append(obs[1])
        metrics['glucose'].append(obs[2])
        metrics['lactate'].append(obs[3])
        metrics['il2'].append(obs[4])
        metrics['exhaustion_fraction'].append(info['exhaustion_fraction'])
        
        if terminated or truncated:
            break
    
    return metrics

def plot_calibration_results(metrics, platform, target_data=None):
    """Plot calibration results against target data"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Calibration Results - {platform.upper()} Platform', fontsize=16)
    
    # Cell growth
    axes[0, 0].plot(metrics['time'], metrics['viable_cells'], 'g-', label='Viable', linewidth=2)
    axes[0, 0].plot(metrics['time'], metrics['exhausted_cells'], 'r-', label='Exhausted', linewidth=2)
    if target_data and 'cell_counts' in target_data:
        axes[0, 0].scatter(target_data['days'], target_data['cell_counts'], 
                          color='blue', s=50, label='Target Data', zorder=5)
    axes[0, 0].set_title('Cell Growth')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Cells (×10⁶)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Nutrients
    axes[0, 1].plot(metrics['time'], metrics['glucose'], 'b-', label='Glucose', linewidth=2)
    axes[0, 1].plot(metrics['time'], metrics['lactate'], 'orange', label='Lactate', linewidth=2)
    if target_data and 'glucose' in target_data:
        axes[0, 1].scatter(target_data['days'], target_data['glucose'], 
                          color='blue', s=50, label='Target Glucose', zorder=5)
    if target_data and 'lactate' in target_data:
        axes[0, 1].scatter(target_data['days'], target_data['lactate'], 
                          color='red', s=50, label='Target Lactate', zorder=5)
    axes[0, 1].set_title('Nutrient Levels')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Concentration (mM)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Exhaustion
    axes[1, 0].plot(metrics['time'], [x*100 for x in metrics['exhaustion_fraction']], 'r-', linewidth=2)
    if target_data and 'exhaustion' in target_data:
        axes[1, 0].scatter(target_data['days'], [x*100 for x in target_data['exhaustion']], 
                          color='blue', s=50, label='Target Exhaustion', zorder=5)
    axes[1, 0].set_title('Exhaustion Fraction')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Exhaustion (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # IL-2
    axes[1, 1].plot(metrics['time'], metrics['il2'], 'purple', linewidth=2)
    if target_data and 'il2' in target_data:
        axes[1, 1].scatter(target_data['days'], target_data['il2'], 
                          color='blue', s=50, label='Target IL-2', zorder=5)
    axes[1, 1].set_title('IL-2 Levels')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('IL-2 (units)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{platform}_calibration.png', dpi=150, bbox_inches='tight')
    plt.show()

def suggest_parameter_adjustments(metrics, target_data, platform):
    """Suggest parameter adjustments based on target data"""
    
    print(f"\n{'='*60}")
    print(f"PARAMETER ADJUSTMENT SUGGESTIONS - {platform.upper()}")
    print(f"{'='*60}")
    
    # Analyze cell growth
    final_viable = metrics['viable_cells'][-1]
    if target_data and 'cell_counts' in target_data:
        target_final = target_data['cell_counts'][-1]
        expansion_ratio = final_viable / 2.0  # Starting with 2M cells
        target_expansion = target_final / 2.0
        
        print(f"CELL GROWTH ANALYSIS:")
        print(f"  Current final cells: {final_viable:.1f}M")
        print(f"  Target final cells: {target_final:.1f}M")
        print(f"  Current expansion: {expansion_ratio:.1f}x")
        print(f"  Target expansion: {target_expansion:.1f}x")
        
        if expansion_ratio < target_expansion * 0.8:
            print(f"  SUGGESTION: Increase r_base (current: {PLATFORMS[platform]['r_base']:.3f})")
            print(f"             Increase K (current: {PLATFORMS[platform]['K']:.0f})")
        elif expansion_ratio > target_expansion * 1.2:
            print(f"  SUGGESTION: Decrease r_base (current: {PLATFORMS[platform]['r_base']:.3f})")
            print(f"             Decrease K (current: {PLATFORMS[platform]['K']:.0f})")
    
    # Analyze exhaustion
    final_exhaustion = metrics['exhaustion_fraction'][-1]
    if target_data and 'exhaustion' in target_data:
        target_exhaustion = target_data['exhaustion'][-1]
        
        print(f"\nEXHAUSTION ANALYSIS:")
        print(f"  Current exhaustion: {final_exhaustion*100:.1f}%")
        print(f"  Target exhaustion: {target_exhaustion*100:.1f}%")
        
        if final_exhaustion > target_exhaustion * 1.2:
            print(f"  SUGGESTION: Decrease exhaustion_base (current: {PLATFORMS[platform]['exhaustion_base']:.4f})")
            print(f"             Decrease exhaustion_activation (current: {PLATFORMS[platform]['exhaustion_activation']:.4f})")
        elif final_exhaustion < target_exhaustion * 0.8:
            print(f"  SUGGESTION: Increase exhaustion_base (current: {PLATFORMS[platform]['exhaustion_base']:.4f})")
            print(f"             Increase exhaustion_activation (current: {PLATFORMS[platform]['exhaustion_activation']:.4f})")
    
    # Analyze nutrients
    min_glucose = min(metrics['glucose'])
    max_lactate = max(metrics['lactate'])
    
    print(f"\nNUTRIENT ANALYSIS:")
    print(f"  Min glucose: {min_glucose:.1f} mM")
    print(f"  Max lactate: {max_lactate:.1f} mM")
    
    if min_glucose < 5.0:
        print(f"  SUGGESTION: Decrease glc_consume (current: {PLATFORMS[platform]['glc_consume']:.3f})")
    if max_lactate > 20.0:
        print(f"  SUGGESTION: Decrease lac_prod (current: {PLATFORMS[platform]['lac_prod']:.3f})")

def create_calibration_template():
    """Create a template for users to input their experimental data"""
    
    template = """
# CAR T-cell Digital Twin Calibration Template
# Fill in your experimental data below

# Example data structure:
target_data = {
    'days': [0, 4, 7, 10, 12],  # Time points
    'cell_counts': [2.0, 15.0, 45.0, 80.0, 100.0],  # Viable cell counts (×10⁶)
    'glucose': [25.0, 20.0, 15.0, 10.0, 8.0],  # Glucose concentrations (mM)
    'lactate': [2.0, 5.0, 10.0, 15.0, 18.0],  # Lactate concentrations (mM)
    'exhaustion': [0.05, 0.08, 0.12, 0.15, 0.18],  # Exhaustion fractions
    'il2': [0, 50, 100, 80, 60]  # IL-2 levels (units)
}

# Platform-specific targets:
# G-Rex: 50-100x expansion, <15% exhaustion
# Stirred Tank: 100-200x expansion, <20% exhaustion

# Typical seed densities:
# G-Rex: 0.5-2.0 × 10⁶ cells/mL
# Stirred Tank: 0.5-1.0 × 10⁶ cells/mL

# Typical IL-2 schedules:
# Day 0-3: 50-100 IU/mL
# Day 4-7: 25-50 IU/mL  
# Day 8+: 0-25 IU/mL
"""
    
    with open('calibration_template.py', 'w') as f:
        f.write(template)
    
    print("Calibration template created: calibration_template.py")
    print("Fill in your experimental data and run calibration_guide.py")

if __name__ == "__main__":
    print("CAR T-cell Digital Twin Calibration Guide")
    print("=" * 50)
    
    # Create template
    create_calibration_template()
    
    # Run calibration for both platforms
    for platform in ["grex", "stirred_tank"]:
        print(f"\nRunning calibration simulation for {platform}...")
        metrics = run_calibration_simulation(platform=platform)
        plot_calibration_results(metrics, platform)
        suggest_parameter_adjustments(metrics, None, platform)
    
    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print("Next steps:")
    print("1. Fill in calibration_template.py with your experimental data")
    print("2. Modify PLATFORMS parameters in car_t_env.py based on suggestions")
    print("3. Re-run calibration to verify improvements")
    print("4. Train PPO agent with calibrated parameters")
