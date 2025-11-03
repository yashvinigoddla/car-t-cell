# simulation.py
import pygame
import random
from dataclasses import dataclass
from typing import Optional

# --- Constants from the paper (can be tuned) ---
GRID_WIDTH = 50
GRID_HEIGHT = 50
INITIAL_CELL_COUNT = 20
SIMULATION_STEPS = 1600 # 7 days * 24 hours/day * 60 min/hour / 6 min/step

# --- New constants for simulation logic (defaults; can be overridden by cell type params) ---
MOVE_PROBABILITY = 0.8
DEFAULT_CONVERSION_PROBABILITY = 0.1
DEFAULT_BEAD_EXHAUSTION_RATE = 0.01
DEFAULT_NATURAL_EXHAUSTION_RATE = 0.001
DEFAULT_PROLIFERATION_PROBABILITY = 0.05
DEFAULT_MIN_PROLIFERATION_POTENCY = 0.6
# Age is in simulation steps (6 min/step). 2 days = 480 steps.
PROLIFERATION_AGE_MIN_STEPS = 480
PROLIFERATION_AGE_MAX_STEPS = 960

@dataclass
class CellParams:
    """Parameters governing cell behavior for a given cell type."""
    conversion_probability: float = DEFAULT_CONVERSION_PROBABILITY
    bead_exhaustion_rate: float = DEFAULT_BEAD_EXHAUSTION_RATE
    natural_exhaustion_rate: float = DEFAULT_NATURAL_EXHAUSTION_RATE
    proliferation_probability: float = DEFAULT_PROLIFERATION_PROBABILITY
    min_proliferation_potency: float = DEFAULT_MIN_PROLIFERATION_POTENCY
    asymmetric_reproduction: bool = False  # True: activated -> activated + naive; False: symmetric (activated -> activated + activated)


class Cell:
    """ Represents a single T-cell with its properties. """
    def __init__(self, x, y, is_activated=False, potency=0.0):
        self.x = x
        self.y = y
        self.is_activated = is_activated
        self.potency = potency
        self.age = 0

    def update(self, occupied_spaces, bead_locations, params: CellParams):
        """ Update cell state for one time step. """
        self.age += 1

        # 1. Movement Logic
        if random.random() < MOVE_PROBABILITY:
            possible_moves = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    new_x, new_y = self.x + dx, self.y + dy
                    if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and (new_x, new_y) not in occupied_spaces:
                        possible_moves.append((new_x, new_y))
            if possible_moves:
                self.x, self.y = random.choice(possible_moves)

        # 2. Activation & Exhaustion Logic
        if self.is_activated:
            # Natural exhaustion for all activated cells
            self.potency -= params.natural_exhaustion_rate
            # Bead-induced exhaustion
            if (self.x, self.y) in bead_locations:
                self.potency -= params.bead_exhaustion_rate
            self.potency = max(0, self.potency)
        else:
            # Activation for naive cells
            if (self.x, self.y) in bead_locations and random.random() < params.conversion_probability:
                self.is_activated = True
                self.potency = 1.0

class Simulation:
    """ Manages the entire simulation grid, cells, and beads. """
    def __init__(self, cell_type: Optional[int] = None, params: Optional[CellParams] = None):
        self.cells = []
        self.beads = []
        # Set cell behavior parameters either from preset cell_type or provided params
        if params is not None:
            self.params = params
        else:
            self.params = self._params_for_cell_type(cell_type)
        self._seed_cells()

    def _params_for_cell_type(self, cell_type: Optional[int]) -> CellParams:
        """Return parameter presets for given cell type ID per paper's variations.
        If None or unknown, return defaults.
        """
        # Base/default (Type 1): balanced
        default = CellParams()
        presets = {
            1: CellParams(  # base case
                conversion_probability=0.1,
                bead_exhaustion_rate=0.01,
                natural_exhaustion_rate=0.001,
                proliferation_probability=0.05,
                min_proliferation_potency=0.6,
                asymmetric_reproduction=False,
            ),
            2: CellParams(  # lower exhaustion, lower conversion
                conversion_probability=0.05,
                bead_exhaustion_rate=0.005,
                natural_exhaustion_rate=0.001,
                proliferation_probability=0.05,
                min_proliferation_potency=0.6,
                asymmetric_reproduction=False,
            ),
            3: CellParams(  # higher natural exhaustion, higher reproduction
                conversion_probability=0.1,
                bead_exhaustion_rate=0.01,
                natural_exhaustion_rate=0.005,
                proliferation_probability=0.10,
                min_proliferation_potency=0.6,
                asymmetric_reproduction=False,
            ),
            4: CellParams(  # like 3, asymmetric reproduction
                conversion_probability=0.1,
                bead_exhaustion_rate=0.01,
                natural_exhaustion_rate=0.005,
                proliferation_probability=0.10,
                min_proliferation_potency=0.6,
                asymmetric_reproduction=True,
            ),
            5: CellParams(  # higher reproduction only
                conversion_probability=0.1,
                bead_exhaustion_rate=0.01,
                natural_exhaustion_rate=0.001,
                proliferation_probability=0.10,
                min_proliferation_potency=0.6,
                asymmetric_reproduction=False,
            ),
            6: CellParams(  # higher natural exhaustion, asymmetric
                conversion_probability=0.1,
                bead_exhaustion_rate=0.01,
                natural_exhaustion_rate=0.005,
                proliferation_probability=0.05,
                min_proliferation_potency=0.6,
                asymmetric_reproduction=True,
            ),
        }
        return presets.get(cell_type, default)

    def _seed_cells(self):
        """ Place initial naive T-cells randomly on the grid, ensuring no overlaps. """
        self.cells = []
        occupied_spaces = set()
        while len(self.cells) < INITIAL_CELL_COUNT:
            x, y = random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)
            if (x, y) not in occupied_spaces:
                self.cells.append(Cell(x, y))
                occupied_spaces.add((x,y))

    def add_beads(self, count=10):
        """ Add activation beads to the simulation. """
        for _ in range(count):
            x, y = random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)
            self.beads.append((x, y))

    def remove_beads(self):
        """ Remove all beads from the simulation. """
        self.beads = []

    def run_step(self):
        """ Run one step of the Monte Carlo simulation. """
        occupied_spaces = {(cell.x, cell.y) for cell in self.cells}
        bead_locations = set(self.beads)
        newly_created_cells = []

        # Update each cell
        for cell in self.cells:
            original_pos = (cell.x, cell.y)
            occupied_spaces.remove(original_pos)
            cell.update(occupied_spaces, bead_locations, self.params)
            occupied_spaces.add((cell.x, cell.y))

        # Handle proliferation after all cells have moved
        for cell in self.cells:
            if cell.is_activated and cell.potency > self.params.min_proliferation_potency and \
               PROLIFERATION_AGE_MIN_STEPS <= cell.age <= PROLIFERATION_AGE_MAX_STEPS and \
               random.random() < self.params.proliferation_probability:
                
                # Find an empty spot for the new cell
                possible_spots = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        new_x, new_y = cell.x + dx, cell.y + dy
                        if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and (new_x, new_y) not in occupied_spaces:
                            possible_spots.append((new_x, new_y))
                
                if possible_spots:
                    new_cell_x, new_cell_y = random.choice(possible_spots)
                    new_cell = Cell(new_cell_x, new_cell_y)
                    # Reproduction mode: asymmetric vs symmetric
                    if self.params.asymmetric_reproduction:
                        # Parent remains activated; child is naive
                        new_cell.is_activated = False
                        new_cell.potency = 0.0
                    else:
                        # Symmetric: child is activated with potency similar to parent
                        new_cell.is_activated = True
                        new_cell.potency = cell.potency
                    newly_created_cells.append(new_cell)
                    occupied_spaces.add((new_cell_x, new_cell_y))
                    cell.age = 0 # Parent cell's age resets

        self.cells.extend(newly_created_cells)

    def get_observation(self):
        """ Gathers the current state of the simulation for the RL agent. """
        if not self.cells:
            return {
                "total_cells": 0,
                "num_activated": 0,
                "num_naive": 0,
                "avg_potency": 0,
                "bead_count": len(self.beads),
                "coverage": 0.0,
                "activated_fraction": 0.0,
            }

        num_activated = sum(1 for cell in self.cells if cell.is_activated)
        avg_potency = sum(c.potency for c in self.cells) / len(self.cells)
        coverage = len(self.cells) / float(GRID_WIDTH * GRID_HEIGHT)
        activated_fraction = num_activated / float(len(self.cells))
        
        return {
            "total_cells": len(self.cells),
            "num_activated": num_activated,
            "num_naive": len(self.cells) - num_activated,
            "avg_potency": avg_potency,
            "bead_count": len(self.beads),
            "coverage": coverage,
            "activated_fraction": activated_fraction,
        }

    def reset(self):
        """ Resets the simulation to its initial state. """
        self.__init__()

    def to_json(self):
        """ Serializes the simulation state to a JSON-friendly dictionary. """
        return {
            'cells': [
                {'x': cell.x, 'y': cell.y, 'is_activated': cell.is_activated, 'potency': cell.potency}
                for cell in self.cells
            ],
            'beads': self.beads,
            'metrics': self.get_observation()
        }
