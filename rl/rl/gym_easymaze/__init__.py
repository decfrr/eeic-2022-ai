from gym.envs.registration import register

register(
    id='EasyMaze-v0',
    entry_point='gym_easymaze.envs:EasyMazeEnv',
)
