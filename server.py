from env import GridEnv

env = GridEnv()

def reset():
    return env.reset()

def step(action):
    return env.step(action)