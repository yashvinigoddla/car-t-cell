# environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation import Simulation # Assuming simulation.py is in the same directory

# --- Constants ---
SIMULATION_STEPS = 1600

class CarTCellEnv(gym.Env):
    """ Custom Gymnasium Environment for the CAR T-Cell Simulation. """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, control_interval=32, cell_type=None):
        super().__init__()
        self.simulation = Simulation(cell_type=cell_type)
        self.control_interval = control_interval # Number of sim steps between agent actions
        self.current_step = 0
        self.max_steps = SIMULATION_STEPS // control_interval
        self.previous_avg_potency = 0
        self.previous_potent_cells = 0
        self.previous_action = 2 # Start with 'skip' as previous action

        # Action space: 0: Add beads, 1: Remove beads, 2: Skip
        self.action_space = spaces.Discrete(3)

        # Observation space (tabular): [total_cells, num_activated, avg_potency, bead_count, time_left,
        #                               previous_action, coverage, activated_fraction]
        # Using Box space for continuous/integer values.
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(8,), 
            dtype=np.float32
        )

    def _get_obs(self):
        obs_dict = self.simulation.get_observation()
        time_left = (self.max_steps - self.current_step) / self.max_steps
        return np.array([
            obs_dict['total_cells'],
            obs_dict['num_activated'],
            obs_dict['avg_potency'],
            obs_dict['bead_count'],
            time_left,
            float(self.previous_action),
            obs_dict['coverage'],
            obs_dict['activated_fraction']
        ], dtype=np.float32)

    def _calculate_reward(self, obs_dict):
        """ Calculate reward based on potency change and a late-bead penalty. """
        current_avg_potency = obs_dict['avg_potency']

        # 1. Reward for increasing potency (encourages activation)
        change_in_potency = current_avg_potency - self.previous_avg_potency
        step_reward = max(0, change_in_potency * 100)

        # 2. Time-weighted penalty for having beads on the grid (encourages removal)
        w_bead = 0.05  # Penalty weight hyperparameter
        time_left = (self.max_steps - self.current_step) / self.max_steps
        bead_penalty = w_bead * obs_dict['bead_count'] * (1 - time_left)
        step_reward -= bead_penalty

        # 3. Large terminal reward for final potency (the ultimate goal)
        terminal_add = 0.0
        if self.current_step >= self.max_steps:
            total_sum = sum(c.potency for c in self.simulation.cells)
            terminal_add = total_sum * 10

        reward = step_reward + terminal_add
        self.previous_avg_potency = current_avg_potency
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulation.reset()
        self.current_step = 0
        self.previous_avg_potency = 0
        self.previous_potent_cells = 0
        self.previous_action = 2
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # 1. Apply action
        if action == 0:
            self.simulation.add_beads()
        elif action == 1:
            self.simulation.remove_beads()
        # action == 2 means do nothing

        # 2. Run simulation for the control interval
        for _ in range(self.control_interval):
            self.simulation.run_step()

        self.current_step += 1
        self.previous_action = action

        # 3. Get observation, reward, and termination status
        obs_dict = self.simulation.get_observation()
        observation = self._get_obs()
        reward = self._calculate_reward(obs_dict)
        
        terminated = self.current_step >= self.max_steps
        truncated = False # Not used here, but required by the API
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        # TODO: Implement visualization using Pygame if needed
        pass

    def close(self):
        # pygame.quit() # If using pygame for rendering
        pass
