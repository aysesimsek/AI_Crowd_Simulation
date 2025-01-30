import pygame
from stable_baselines3 import PPO
from grid_environment import FireEvacuationEnv

# Pygame Settings
GRID_SIZE = 20
CELL_SIZE = 30
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

# Colors for visualization
COLORS = {0: (255, 255, 255), 1: (200, 0, 0), 2: (0, 200, 0), 3: (0, 0, 200)}

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Fire Evacuation Simulation")
clock = pygame.time.Clock()

# Initialize the environment
env = FireEvacuationEnv()

# Load trained AI model
model = PPO.load("ppo_fire_evacuation")

running = True
while running:
    screen.fill((255, 255, 255))

    # Draw grid elements
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            pygame.draw.rect(screen, COLORS[env.grid[x, y]], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Get AI-Powered movement for NPCs
    obs = env.grid.copy()
    action, _states = model.predict(obs)  # AI decides movement

    env.step(action)  # ✅ Pass AI-generated action to step()

    # Draw NPCs (Blue Circles)
    for npc in env.npcs:
        pygame.draw.circle(screen, (0, 0, 200), 
                           (npc[0] * CELL_SIZE + CELL_SIZE // 2, npc[1] * CELL_SIZE + CELL_SIZE // 2), 
                           CELL_SIZE // 3)

    pygame.display.flip()
    clock.tick(2)  # Simulation speed

    # Handle Quit Event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
