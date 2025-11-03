# car_t_env.py
# Minimal digital twin + Gymnasium env for CAR-T activation/expansion
# Platforms: "grex" and "stirred_tank"
# Author: you

import math, numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------
# Platform parameter presets
# ---------------------------
PLATFORMS = {
    "grex": dict(
        dt_h=6.0,                  # step size (hours) - finer control
        K=400.0,                   # carrying capacity (x1e6 cells) - increased
        r_base=0.25,               # baseline daily growth rate - more conservative
        o2_bonus=0.12,             # O2 bonus (gas-permeable base) - increased
        shear_penalty=0.00,        # essentially none
        glc_consume=0.20,          # glucose consumption per 6h per 1e8 cells (mM) - reduced
        lac_prod=0.18,             # lactate production per 6h per 1e8 cells (mM) - reduced
        il2_decay=0.15,            # IL-2 decay per step (fraction) - slower decay
        exhaustion_base=0.001,     # baseline exhaustion accrual per step - reduced
        exhaustion_activation=0.015,# extra exhaustion when heavily stimulated - increased
        exhaustion_il2=0.008,      # extra exhaustion when IL2 is high - increased
        glc_init=25.0,             # starting glucose (mM)
        lac_init=2.0,              # starting lactate (mM)
        il2_init=0.0,              # starting IL-2
        Nv_init=2.0,               # start viable (x1e6)
        Ne_init=0.1,               # start exhausted (x1e6)
        vol_ml=300.0               # working volume (mL)
    ),
    "stirred_tank": dict(
        dt_h=6.0,                  # step size (hours) - finer control
        K=800.0,                   # higher effective capacity with feeding/oxygenation - increased
        r_base=0.30,               # slightly higher growth potential - tuned
        o2_bonus=0.15,             # more O2 with agitation - increased
        shear_penalty=0.08,        # too much agitation harms growth - increased
        glc_consume=0.25,          # glucose consumption per 6h per 1e8 cells (mM) - reduced
        lac_prod=0.22,             # lactate production per 6h per 1e8 cells (mM) - reduced
        il2_decay=0.18,            # IL-2 decay per step (fraction) - slower decay
        exhaustion_base=0.001,     # baseline exhaustion accrual per step - reduced
        exhaustion_activation=0.015,# extra exhaustion when heavily stimulated - increased
        exhaustion_il2=0.008,      # extra exhaustion when IL2 is high - increased
        glc_init=25.0,             # starting glucose (mM)
        lac_init=2.0,              # starting lactate (mM)
        il2_init=0.0,              # starting IL-2
        Nv_init=2.0,               # start viable (x1e6)
        Ne_init=0.1,               # start exhausted (x1e6)
        vol_ml=500.0               # working volume (mL)
    )
}

# Utility clipping
def clamp(x, lo, hi): return max(lo, min(hi, x))

class CarTManufactureEnv(gym.Env):
    """
    A simple CAR-T activation/expansion environment.

    Observation (continuous):
        [Nv, Ne, Glu, Lac, IL2, B, age_act, platform_id_onehot(2)]
    Action (MultiDiscrete):
        0: beads {0:none,1:add_small,2:add_large,3:remove_all}
        1: IL2   {0:none,1:low,2:med,3:high}
        2: media {0:0%, 1:25%, 2:50%, 3:100%}
        3: agitation (only used in stirred-tank) {0:low,1:med,2:high} (ignored for G-Rex)
    """
    metadata = {"render_modes": []}

    def __init__(self, platform="grex", episode_days=12, seed=None):
        super().__init__()
        assert platform in PLATFORMS, f"platform must be one of {list(PLATFORMS)}"
        self.pname = platform
        self.p = PLATFORMS[platform].copy()
        self.rng = np.random.default_rng(seed)
        self.steps = int((episode_days*24.0) / self.p["dt_h"])

        # Observation space (normalized-ish ranges)
        # Nv,Ne up to ~K; Glu 0-30mM; Lac 0-30mM; IL2 0-1000 (arbitrary units)
        low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high= np.array([self.p["K"], self.p["K"], 40, 40, 1000, 1, 240, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Actions
        self.action_space = spaces.MultiDiscrete([4, 4, 4, 3])  # beads, IL2, media, agitation

        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p = self.p
        self.t = 0
        # states
        self.Nv = max(0.5, p["Nv_init"] + self.rng.normal(0, 0.2))
        self.Ne = max(0.0, p["Ne_init"] + self.rng.normal(0, 0.05))
        self.Glu = p["glc_init"]
        self.Lac = p["lac_init"]
        self.IL2 = p["il2_init"]
        self.B = 0.0         # stimulation occupancy 0..1
        self.age_act = 999.0 # large means not recently activated
        self.agitation = 0   # 0/1/2 (ignored in grex)
        obs = self._obs()
        return obs, {}

    def _obs(self):
        plat_onehot = np.array([1,0]) if self.pname=="grex" else np.array([0,1])
        return np.array([
            self.Nv, self.Ne, self.Glu, self.Lac, self.IL2, self.B, self.age_act,
            *plat_onehot
        ], dtype=np.float32)

    def step(self, action):
        beads_a, il2_a, media_a, ag_a = action
        day = self.t * self.p["dt_h"] / 24.0
        
        # --- Apply action limits to prevent abuse ---
        # Limit bead additions after day 8 (prevent over-stimulation)
        if day > 8.0 and beads_a in (1, 2):
            beads_a = 0  # Force no bead action after day 8
        
        # Limit IL-2 additions after day 6 (prevent excessive cytokine)
        if day > 6.0 and il2_a in (2, 3):
            il2_a = 1  # Force low IL-2 only after day 6
        
        # --- Apply actions (bounded, with simple effects) ---
        # Beads
        if beads_a == 1:   # add small
            self.B = clamp(self.B + 0.35, 0.0, 1.0); self.age_act = 0.0
        elif beads_a == 2: # add large
            self.B = clamp(self.B + 0.65, 0.0, 1.0); self.age_act = 0.0
        elif beads_a == 3: # remove all
            self.B = 0.0

        # IL-2 pulse (arbitrary units) - with daily limits
        daily_il2_limit = 500  # Maximum IL-2 per day
        if il2_a == 1: 
            self.IL2 = min(self.IL2 + 50, daily_il2_limit)
        elif il2_a == 2: 
            self.IL2 = min(self.IL2 + 150, daily_il2_limit)
        elif il2_a == 3: 
            self.IL2 = min(self.IL2 + 300, daily_il2_limit)

        # Media refresh (% replaces medium → resets Glu/Lac proportionally)
        if media_a in (1,2,3):
            frac = [0.0, 0.25, 0.50, 1.0][media_a]
            self.Glu = self.Glu*(1-frac) + PLATFORMS[self.pname]["glc_init"]*frac
            self.Lac = self.Lac*(1-frac)
            # IL-2 also dilutes
            self.IL2 = self.IL2*(1-frac)

        # Agitation (only matters in stirred_tank)
        self.agitation = ag_a if self.pname=="stirred_tank" else 0

        # --- Dynamics (one Euler step over dt_h) ---
        p = self.p
        dt = p["dt_h"] / 24.0  # convert hours→days for rates

        # Effective growth rate r_eff
        r = p["r_base"]
        # oxygen bonus: grex fixed; stirred-tank depends on agitation
        if self.pname == "grex":
            r += p["o2_bonus"]
        else:
            r += p["o2_bonus"] * (0.5 + 0.5*self.agitation/2.0)  # low=0.5x .. high=1.0x
            r -= p["shear_penalty"] * (self.agitation/2.0)       # shear penalty

        # stimulation boosts division early, then plateaus/harms via exhaustion accrual
        stim_boost = 1.0 + 0.6*self.B * math.exp(-self.age_act/36.0)  # decays with age since stim
        r *= stim_boost

        # nutrient limitation (Monod-like): Glu effect ∈ [0.1,1.0]
        glc_factor = clamp(self.Glu/5.0, 0.1, 1.0)  # ~linear proxy; tune
        r *= glc_factor

        # lactate toxicity reduces r (soft penalty)
        lac_factor = clamp(1.0 - (self.Lac/30.0), 0.0, 1.0)
        r *= lac_factor

        # logistic crowding
        Ntot = max(1e-6, self.Nv + self.Ne)
        crowd = clamp(1.0 - (Ntot/p["K"]), 0.0, 1.0)
        r *= (0.2 + 0.8*crowd)  # never zero; slows near K

        # Growth & phenotypic flows
        dNv_growth = self.Nv * r * dt
        # Exhaustion accrual (moves from Nv→Ne)
        ex = p["exhaustion_base"]
        ex += p["exhaustion_activation"] * self.B * (1.0/(1.0+math.exp(-(self.age_act-12)/8.0))) # worse if beads on for long
        ex += p["exhaustion_il2"] * clamp(self.IL2/300.0, 0.0, 1.0)
        dEx = self.Nv * ex * dt

        # Deaths (very simple): increase with high Lac & starvation
        death_rate = 0.01 + 0.08*(1.0-lac_factor) + 0.05*(1.0-glc_factor)
        dNv_death = self.Nv * death_rate * dt
        dNe_death = self.Ne * (death_rate*1.2) * dt

        # Update populations
        self.Nv = clamp(self.Nv + dNv_growth - dEx - dNv_death, 0.0, p["K"])
        self.Ne = clamp(self.Ne + dEx - dNe_death, 0.0, p["K"])

        # Nutrient/cytokine dynamics
        # Consumption/production scaled by total cells (per 1e8 cells factor)
        scale = (Ntot / 100.0)  # 100 x1e6 = 1e8
        self.Glu = clamp(self.Glu - p["glc_consume"]*scale, 0.0, 40.0)
        self.Lac = clamp(self.Lac + p["lac_prod"]*scale, 0.0, 40.0)
        self.IL2 = clamp(self.IL2 * (1.0 - p["il2_decay"]), 0.0, 1000.0)

        # Age since activation increases if beads present; else still increases but slower harm
        self.age_act = clamp(self.age_act + p["dt_h"], 0.0, 240.0)

        # Small process noise (unobserved variability)
        self.Nv = max(0.0, self.Nv + self.rng.normal(0, 0.01))
        self.Ne = max(0.0, self.Ne + self.rng.normal(0, 0.01))

        # Reward: prioritize viable yield & low exhaustion at the END; give shaping mid-episode
        done = (self.t >= self.steps-1)
        ex_frac = (self.Ne / max(1e-6, self.Nv + self.Ne))
        step_cost = 0.0
        # cost for resource use
        step_cost -= 0.0002 * (il2_a*50)              # IL-2 penalty
        step_cost -= 0.02   * ([0,0.25,0.5,1.0][media_a])  # media usage
        step_cost -= 0.01   * (beads_a in (1,2))      # bead use
        # small shaping: prefer low lac, non-starvation
        step_shaping = 0.01 * (self.Glu/25.0) + 0.01 * (1.0 - ex_frac) - 0.01*(self.Lac/20.0)

        if done:
            reward = ( + 1.0 * (self.Nv / (0.5*self.p["K"]))          # normalize to K
                        - 1.2 * ex_frac
                        - 0.2 * (self.Lac/30.0)
                        + step_cost + step_shaping )
        else:
            reward = step_cost + step_shaping

        self.t += 1
        obs = self._obs()
        info = {"day": self.t * self.p["dt_h"]/24.0, "exhaustion_fraction": float(ex_frac)}
        terminated, truncated = done, False
        return obs, float(reward), terminated, truncated, info

# ------------- quick smoke test -------------
if __name__ == "__main__":
    for plat in ("grex","stirred_tank"):
        env = CarTManufactureEnv(platform=plat, episode_days=12, seed=42)
        obs,_ = env.reset()
        total = 0.0
        for _ in range(env.steps):
            # naive heuristic: if Glu low or Lac high → refresh; pulse IL-2 early; add beads day 0/3
            Nv,Ne,Glu,Lac,IL2,B,age, p0, p1 = obs
            day = _ * env.p["dt_h"]/24.0
            beads = 1 if day<0.5 else (1 if abs(day-3.0)<0.1 else 0)  # small add day 0 and ~day3
            il2   = 2 if day<2.0 else 1 if day<6.0 else 0
            media = 3 if (Glu<8 or Lac>18) else 0
            ag    = 1  # medium agitation (ignored in grex)
            action = np.array([beads, il2, media, ag])
            obs, r, term, trunc, info = env.step(action)
            total += r
            if term: break
        print(f"{plat}: return={total:.3f}, Nv={obs[0]:.2f}, Ne={obs[1]:.2f}, ex%={100*info['exhaustion_fraction']:.1f}")
