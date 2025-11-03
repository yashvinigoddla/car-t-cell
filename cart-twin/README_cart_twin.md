# CAR T-cell Digital Twin Environment

A Gymnasium-compatible environment for simulating CAR T-cell manufacturing processes with reinforcement learning optimization.

## Overview

This digital twin simulates the CAR T-cell activation and expansion process in two different bioreactor platforms:
- **G-Rex**: Gas-permeable base, low shear, smaller capacity
- **Stirred Tank**: Higher capacity with agitation control, better mixing but shear sensitivity

## Files

- `car_t_env.py` - Main environment implementation
- `demo_cart_env.py` - Demo script showing environment usage
- `train_cart_agent.py` - Training script with simple Q-learning agent
- `requirements_cart.txt` - Python dependencies

## Installation

```bash
pip install -r requirements_cart.txt
```

## Quick Start

### 1. Run the Demo
```bash
python demo_cart_env.py
```
This will run a naive heuristic strategy on both platforms and show comparison results.

### 2. Train an RL Agent
```bash
python train_cart_agent.py
```
This will train a simple Q-learning agent on both platforms and show training progress.

### 3. Use the Environment Directly
```python
from car_t_env import CarTManufactureEnv

# Create environment
env = CarTManufactureEnv(platform="grex", episode_days=12, seed=42)

# Reset and run
obs, _ = env.reset()
for step in range(env.steps):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Environment Details

### State Space (9 dimensions)
- `Nv`: Viable cells (×10⁶)
- `Ne`: Exhausted cells (×10⁶) 
- `Glu`: Glucose concentration (mM)
- `Lac`: Lactate concentration (mM)
- `IL2`: IL-2 cytokine level (units)
- `B`: Bead occupancy (0-1)
- `age_act`: Hours since last activation
- `platform_id`: One-hot encoding of platform type

### Action Space (4 discrete actions)
- **Beads**: {0:none, 1:add_small, 2:add_large, 3:remove_all}
- **IL-2**: {0:none, 1:low, 2:med, 3:high}
- **Media**: {0:0%, 1:25%, 2:50%, 3:100%}
- **Agitation**: {0:low, 1:med, 2:high} (stirred_tank only)

### Reward Function
The reward balances multiple objectives:
- **Primary**: Maximize final viable cell count
- **Secondary**: Minimize exhaustion fraction
- **Costs**: Penalize resource usage (IL-2, media, beads)
- **Safety**: Penalize high lactate and starvation

### Platform Differences

| Feature | G-Rex | Stirred Tank |
|---------|-------|--------------|
| Capacity | 300×10⁶ cells | 600×10⁶ cells |
| O₂ Supply | Gas-permeable base | Agitation-dependent |
| Shear | Minimal | Variable (agitation-dependent) |
| Growth Rate | 0.30/day | 0.35/day |
| Volume | 300 mL | 500 mL |

## Key Biological Processes Modeled

1. **Cell Growth**: Logistic growth with nutrient and oxygen limitations
2. **Exhaustion**: Progressive loss of function due to overstimulation
3. **Nutrient Dynamics**: Glucose consumption and lactate production
4. **Cytokine Effects**: IL-2 promotes growth but can increase exhaustion
5. **Stimulation Effects**: Bead activation boosts growth initially, then causes exhaustion

## Usage Examples

### Custom Strategy
```python
def my_strategy(obs, day):
    Nv, Ne, Glu, Lac, IL2, B, age, p0, p1 = obs
    
    # Add beads early in process
    if day < 1.0:
        beads = 1
    elif day > 6.0 and B > 0.5:
        beads = 3  # Remove beads if overstimulated
    else:
        beads = 0
    
    # IL-2 strategy
    if day < 3.0:
        il2 = 2  # Medium IL-2 early
    else:
        il2 = 0
    
    # Media refresh when needed
    if Glu < 10 or Lac > 15:
        media = 3  # Full refresh
    else:
        media = 0
    
    return np.array([beads, il2, media, 1])  # Medium agitation
```

### Training with Stable-Baselines3
```python
from stable_baselines3 import PPO
from car_t_env import CarTManufactureEnv

env = CarTManufactureEnv(platform="grex")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Biological Insights

The environment captures several key aspects of CAR T-cell manufacturing:

1. **Activation-Exhaustion Trade-off**: Early stimulation promotes growth but prolonged exposure causes exhaustion
2. **Nutrient Management**: Glucose depletion and lactate accumulation limit growth
3. **Platform Optimization**: Different bioreactor designs require different strategies
4. **Resource Efficiency**: Balancing cell yield with process costs

## Future Extensions

- Add more sophisticated cell state models
- Include patient-specific variability
- Model additional cytokines and growth factors
- Add contamination and sterility constraints
- Implement multi-objective optimization

## References

Based on research from:
- Ferdous et al. (2023) - RL-guided CAR T-cell activation
- Ledergor et al. (2024) - CD4+ CAR T-cell exhaustion in multiple myeloma
- Sayadmanesh et al. (2023) - aAPC characterization for CAR T manufacturing
