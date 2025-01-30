import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

# Grid size
GRID_SIZE = 20
EXIT_POINTS = [(0, GRID_SIZE // 2), (GRID_SIZE - 1, GRID_SIZE // 2)]
FIRE_START = (GRID_SIZE // 2, GRID_SIZE // 2)
NPC_COUNT = 10

class FireEvacuationEnv(gym.Env):
    def __init__(self):
        super(FireEvacuationEnv, self).__init__()

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=0, high=3, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # 4 possible moves (up, down, left, right)

        # Initialize the grid
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for ex in EXIT_POINTS:
            self.grid[ex] = 2  # Mark exits

        self.fire_positions = [FIRE_START]
        self.grid[FIRE_START] = 1  # Mark fire

        # Initialize NPCs at random locations
        self.npcs = [(random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2)) for _ in range(NPC_COUNT)]

    def step(self, action):
        """Process one step in the environment"""
    
        # Move NPCs based on AI-chosen action
        self.move_npcs(action)

        # Spread fire
        self.spread_fire()

        # Calculate reward (if NPC reaches exit)
        reward = 0
        done = False
        truncated = False  # New variable added (for Gym compatibility)

        for npc in self.npcs:
            if self.grid[npc] == 2:  # NPC reached exit
                reward += 10
                done = True

        return self.grid.copy(), reward, done, truncated, {}  # Returning 5 values


    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)  # Ensure compatibility with Stable-Baselines3
        self.__init__()
        return self.grid.copy(), {}  # Return observation + info dictionary

    def spread_fire(self):
        """Fire spreads dynamically based on wind & obstacles"""
        new_fires = []
        
        wind_direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])  # Wind can push fire in a direction

        for fx, fy in self.fire_positions:
            for dx, dy in [wind_direction, (-wind_direction[0], -wind_direction[1]), (0, 0)]:  # Favor wind direction
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[nx, ny] == 0:
                    new_fires.append((nx, ny))
                    self.grid[nx, ny] = 1  # Mark as fire

        self.fire_positions.extend(new_fires)

    def move_npcs(self, action):
        """NPCs take smarter actions based on AI decisions"""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Up, Down
        
        new_npcs = []
        for x, y in self.npcs:
            possible_moves = [(x + dx, y + dy) for dx, dy in directions]

            # Filter only safe moves (avoid fire)
            safe_moves = [m for m in possible_moves if 0 <= m[0] < GRID_SIZE and 0 <= m[1] < GRID_SIZE and self.grid[m] != 1]

            if safe_moves:
                # AI selects the best move among safe moves
                best_move = min(safe_moves, key=lambda m: min(abs(m[0] - ex[0]) + abs(m[1] - ex[1]) for ex in EXIT_POINTS))
                new_npcs.append(best_move)
            else:
                new_npcs.append((x, y))  # Stay in place if no safe move

        self.npcs = new_npcs

