import numpy as np


class Agent:
    def __init__(self, env, n_actions):

        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 5e-5
        self.n_actions = n_actions
        self.max_steps = 100000
        self.max_episodes = 2000000
        self.env = env

    def choose_action(self, step):

        exp = np.random.rand()
        exp_probability = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-self.decay_rate * step)

        if exp < exp_probability:
            print(f"epsilon:{exp:0.3f}")
            return np.random.randint(self.n_actions)
        else:
            print(f"epsilon:{exp:0.3f}, epsilon threshold:{exp_probability:0.3f}")
            print("Exploration finished !")
            self.env.close()
            exit(0)

    def train(self):
        step = 0
        self.env.reset()
        for episode in range(self.max_episodes):
           step += 1
           action = self.choose_action(step)
           next_state, reward, done, _, = self.env.step(action)
           self.env.render()
           if done:
               self.env.reset()

           print(f"step:{step}")
