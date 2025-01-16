import gymnasium as gym 
from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import make_vec_env


models_dir = "models/PPO"

model_path = f"{models_dir}/400000.zip"


vec_env =make_vec_env("LunarLander-v2")
vec_env.reset()


model = PPO.load(model_path, env=vec_env)


episodes = 10 

for ep in range(episodes):
  obs = vec_env.reset()
  terminated = False
  while not terminated:
    vec_env.render()
    action, _sates = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    
    
    
vec_env.close()