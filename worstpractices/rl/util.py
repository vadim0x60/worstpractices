import gym

class GoOutWithABang(gym.Wrapper):
    def __init__(self, env, bang):
        super().__init__(env)
        self.bang = bang

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            self.bang(observation, reward, done, info)
        return observation, reward, done, info

class LinearLogger():
    def __init__(self, logger):
        self.logger = logger
        self.steps = {}

    def write(self, step_type, data):
        try:
            self.steps[step_type] += 1
        except KeyError:
            self.steps[step_type] = 1

        self.logger.write(step_type, self.steps[step_type], data)