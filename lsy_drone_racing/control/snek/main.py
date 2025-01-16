from snek_env import SnekEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env 

def main():
  #Create environment (no rendering for training)
  env = SnekEnv(render_mode =None )
  check_env(env, warn=True) # Ensure envireonment is valide 
  
  # Train a PPO model for demonstration 
  model = PPO("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps =10_000)
  
  
  # Enjoy the trainded agent 
  env = SnekEnv(render_mode = "human")
  obs, info = env.reset()
  for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
      obs, info = env.reset()
    
    env.close()
    

if __name__ == "__main__":
  main()