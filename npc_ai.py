import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO  # ✅ Import RecurrentPPO
from grid_environment import FireEvacuationEnv

def train_ai():
    """Train AI NPCs using Reinforcement Learning (PPO with LSTM)"""
    env = FireEvacuationEnv()
    vec_env = DummyVecEnv([lambda: env])

    model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1)  # ✅ Use correct policy
    model.learn(total_timesteps=500000)  # Train longer

    model.save("ppo_fire_evacuation")  # Save trained model
    print("Training complete! Model saved as ppo_fire_evacuation.zip")

if __name__ == '__main__':
    train_ai()
