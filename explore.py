import gymnasium as gym
import ale_py


gym.register_envs(ale_py)
env = gym.make('ALE/Boxing-v5', render_mode='human') 
env.reset()

# Enable screen display and sound output
ale = ale_py.ALEInterface()
ale.setBool('display_screen', True)
ale.setBool('sound', True)

for _ in range(1000):
    action = env.action_space.sample()
    print(env.action_space)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        
    break

env.close()