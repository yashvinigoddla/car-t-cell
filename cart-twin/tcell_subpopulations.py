#!/usr/bin/env python3
"""
T-cell subpopulation modeling for CAR T-cell digital twin
Adds Tcm/Tn/Tef compartments with different transition rates
"""

import numpy as np
from car_t_env import CarTManufactureEnv, PLATFORMS

class TCellSubpopulationEnv(CarTManufactureEnv):
    """
    Extended CAR T-cell environment with T-cell subpopulations
    """
    
    def __init__(self, platform="grex", episode_days=12, seed=None):
        super().__init__(platform, episode_days, seed)
        
        # T-cell subpopulations
        self.Tn = 0.0    # Naive T cells
        self.Tcm = 0.0   # Central memory T cells
        self.Tef = 0.0   # Effector T cells
        self.Tex = 0.0   # Exhausted T cells
        
        # Transition rates
        self.transition_rates = {
            'Tn_to_Tcm': 0.1,    # Naive to central memory
            'Tn_to_Tef': 0.05,   # Naive to effector
            'Tcm_to_Tef': 0.15,  # Central memory to effector
            'Tef_to_Tex': 0.02,  # Effector to exhausted
            'Tcm_to_Tex': 0.01,  # Central memory to exhausted
            'Tex_recovery': 0.005 # Exhausted recovery (very slow)
        }
        
        # Update observation space to include subpopulations
        low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([self.p["K"], self.p["K"], 40, 40, 1000, 1, 240, 1, 1, 
                        self.p["K"], self.p["K"], self.p["K"]], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        p = self.p
        self.t = 0
        
        # Initialize subpopulations
        self.Tn = max(0.5, p["Nv_init"] * 0.8)  # 80% naive
        self.Tcm = max(0.1, p["Nv_init"] * 0.15)  # 15% central memory
        self.Tef = max(0.05, p["Nv_init"] * 0.05)  # 5% effector
        self.Tex = max(0.0, p["Ne_init"])  # Exhausted
        
        # Other states
        self.Glu = p["glc_init"]
        self.Lac = p["lac_init"]
        self.IL2 = p["il2_init"]
        self.B = 0.0
        self.age_act = 999.0
        self.agitation = 0
        
        obs = self._obs()
        return obs, {}
    
    def _obs(self):
        """Extended observation with subpopulations"""
        plat_onehot = np.array([1,0]) if self.pname=="grex" else np.array([0,1])
        return np.array([
            self.Tn, self.Tcm, self.Tef, self.Tex,  # T-cell subpopulations
            self.Glu, self.Lac, self.IL2, self.B, self.age_act,
            *plat_onehot
        ], dtype=np.float32)
    
    def step(self, action):
        """Extended step with subpopulation dynamics"""
        
        # Run parent step first
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update subpopulations
        self._update_subpopulations()
        
        # Calculate memory-like pool (Tcm + Tef)
        memory_pool = self.Tcm + self.Tef
        
        # Enhanced reward based on memory pool
        if terminated:
            memory_bonus = 0.5 * (memory_pool / (0.3 * self.p["K"]))  # Reward memory cells
            reward += memory_bonus
        
        # Update info
        info['memory_pool'] = memory_pool
        info['Tn'] = self.Tn
        info['Tcm'] = self.Tcm
        info['Tef'] = self.Tef
        info['Tex'] = self.Tex
        
        return self._obs(), reward, terminated, truncated, info
    
    def _update_subpopulations(self):
        """Update T-cell subpopulation dynamics"""
        
        dt = self.p["dt_h"] / 24.0  # Convert to days
        
        # Calculate transition probabilities based on stimulation and IL-2
        stim_factor = 1.0 + 0.5 * self.B  # Stimulation increases transitions
        il2_factor = 1.0 + 0.3 * min(self.IL2 / 300.0, 1.0)  # IL-2 promotes activation
        
        # Transitions
        Tn_to_Tcm = self.Tn * self.transition_rates['Tn_to_Tcm'] * stim_factor * dt
        Tn_to_Tef = self.Tn * self.transition_rates['Tn_to_Tef'] * stim_factor * dt
        Tcm_to_Tef = self.Tcm * self.transition_rates['Tcm_to_Tef'] * il2_factor * dt
        Tef_to_Tex = self.Tef * self.transition_rates['Tef_to_Tex'] * (1.0 + 0.5 * self.B) * dt
        Tcm_to_Tex = self.Tcm * self.transition_rates['Tcm_to_Tex'] * (1.0 + 0.3 * self.B) * dt
        Tex_recovery = self.Tex * self.transition_rates['Tex_recovery'] * dt
        
        # Update populations
        self.Tn = max(0.0, self.Tn - Tn_to_Tcm - Tn_to_Tef)
        self.Tcm = max(0.0, self.Tcm + Tn_to_Tcm - Tcm_to_Tef - Tcm_to_Tex + Tex_recovery)
        self.Tef = max(0.0, self.Tef + Tn_to_Tef + Tcm_to_Tef - Tef_to_Tex)
        self.Tex = max(0.0, self.Tex + Tef_to_Tex + Tcm_to_Tex - Tex_recovery)
        
        # Update parent class variables for compatibility
        self.Nv = self.Tn + self.Tcm + self.Tef
        self.Ne = self.Tex

def create_subpopulation_reward_function():
    """Create a reward function that prioritizes memory-like cells"""
    
    def subpopulation_reward(env, action, obs, reward, terminated, truncated, info):
        """Enhanced reward function for subpopulations"""
        
        if terminated:
            # Base reward from parent
            base_reward = reward
            
            # Memory pool bonus
            memory_pool = info.get('memory_pool', 0)
            memory_bonus = 0.3 * (memory_pool / (0.3 * env.p["K"]))
            
            # Exhaustion penalty
            exhaustion_penalty = 0.5 * info.get('exhaustion_fraction', 0)
            
            # Phenotype diversity bonus (prefer balanced populations)
            Tn, Tcm, Tef, Tex = obs[0], obs[1], obs[2], obs[3]
            total_cells = Tn + Tcm + Tef + Tex
            if total_cells > 0:
                diversity_bonus = 0.1 * (1.0 - abs(0.3 - Tcm/total_cells) - abs(0.4 - Tef/total_cells))
            else:
                diversity_bonus = 0.0
            
            return base_reward + memory_bonus - exhaustion_penalty + diversity_bonus
        
        return reward
    
    return subpopulation_reward

def test_subpopulation_env():
    """Test the subpopulation environment"""
    
    print("Testing T-cell subpopulation environment...")
    
    env = TCellSubpopulationEnv(platform="grex", episode_days=12, seed=42)
    obs, _ = env.reset()
    
    print(f"Initial subpopulations:")
    print(f"  Naive (Tn): {obs[0]:.1f}")
    print(f"  Central Memory (Tcm): {obs[1]:.1f}")
    print(f"  Effector (Tef): {obs[2]:.1f}")
    print(f"  Exhausted (Tex): {obs[3]:.1f}")
    
    total_reward = 0
    step_count = 0
    
    for step in range(env.steps):
        # Simple strategy
        day = step * env.p["dt_h"] / 24.0
        
        if day < 1.0:
            beads = 1
        else:
            beads = 0
        
        if day < 3.0:
            il2 = 1
        else:
            il2 = 0
        
        media = 0
        agitation = 1
        
        action = np.array([beads, il2, media, agitation])
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        if step % 4 == 0:  # Print every 4 steps
            print(f"Day {day:.1f}: Tn={obs[0]:.1f}, Tcm={obs[1]:.1f}, Tef={obs[2]:.1f}, Tex={obs[3]:.1f}")
        
        if terminated or truncated:
            break
    
    print(f"\nFinal results:")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Final Tn: {obs[0]:.1f}")
    print(f"  Final Tcm: {obs[1]:.1f}")
    print(f"  Final Tef: {obs[2]:.1f}")
    print(f"  Final Tex: {obs[3]:.1f}")
    print(f"  Memory pool: {info.get('memory_pool', 0):.1f}")
    print(f"  Exhaustion: {info.get('exhaustion_fraction', 0)*100:.1f}%")

if __name__ == "__main__":
    test_subpopulation_env()
