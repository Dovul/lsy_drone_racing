import gymnasium as gym
import numpy as np
from gymnasium import spaces
from snek import BLOCK_SIZE, SnakeGame


class SnekEnv(gym.Env):
    """
    Custom Snake environment following the Gymnasium interface.
    Discrete Action Space:
      0 = Left
      1 = Right
      2 = Up
      3 = Down

    Observation:
      - For demonstration: 2D grid of shape (rows, cols) with:
        0 = empty, 1 = snake, 2 = food
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, w=640, h=480, render_mode=None): #"None" by default 
        super().__init__()
        self.w = w
        self.h = h 
        self.render_mode = render_mode
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        
        #Action Space (Left, Right, Up, Down)
        self.action_space = spaces.Discrete(4)
        
        #Observation Space: rows x cols grid 
        self.cols = self.h //BLOCK_SIZE
        self.rows = self.w //BLOCK_SIZE
        self.observation_space = spaces.Box(low = 0, high = 2, 
                                            shape=(self.rows, self.cols),
                                            dtype=np.int32)
        self.game = None
        self._max_episode_steps = 500 
        self.episode_steps = 0
        

    def step(self, action):
        self.episode_step += 1 
        
        # Let the SnakeGame process the action
        reward, terminated = self.game.play_step(action)
        observation = self._get_observation()
        
        truncated = False
        
        if self.episode_steps >= self._max_episode_steps:
            truncated = True 
            
        info = {"score": self.game.score}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset()
        self.episode_steps = 0
        
        # Create a new SnakeGame each time we reset 
        # If we want to render, we pass enable_render=True
        enable_render = (self.render_mode == "human")
        self.game = SnakeGame(w=self.w, h=self.h, enable_render=enable_render)
        
        observation = self._get_observation()
        info = {}
        return observation, info 
    
        
     

    def render(self):
        # We rely on Pygame rendering if enable_render = True 
        pass
    

    def close(self):
        if self.game:
            self.game.close()
            
    def _get_observation(self):
        """
        Return a 2D grid representation of the game state:
        0 = empty 
        1 = snake 
        2 = food 
        """
        grid = np.zeros((self.rows, self.cols), dtype =np.int32)
        
        #Mark snake positions
        
        for pt in self.game.snake:
            col = int(pt.x // BLOCK_SIZE)
            row = int(pt.y // BLOCK_SIZE)
            if 0 <= row < self.rows and 0 <=col < self.cols:
                grid[row, col] = 1 
                
        # Mark food 
        fx =int(self.game.food.x // BLOCK_SIZE)
        fy = int(self.game.food.y // BLOCK_SIZE)
        
        if 0 <= fy < self.rows and 0 <= fx < self.cosl:
            grid[fy, fx] = 2
        
        return grid 