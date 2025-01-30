import matplotlib.pyplot as plt
import numpy as np
from grid_environment import FireEvacuationEnv

def generate_heatmap(env, num_simulations=50):
    """Tracks NPC movement & generates a heatmap"""
    
    heatmap = np.zeros((GRID_SIZE, GRID_SIZE))

    for _ in range(num_simulations):
        env.reset()
        for _ in range(30):  # Simulate 30 steps per run
            action = np.random.randint(0, 4)  # Random action (for testing)
            env.step(action)
            
            for npc in env.npcs:
                heatmap[npc] += 1  # Track NPC presence

    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title("NPC Movement Heatmap")
    plt.colorbar()
    plt.show()

# Run the heatmap generator
env = FireEvacuationEnv()
generate_heatmap(env)
