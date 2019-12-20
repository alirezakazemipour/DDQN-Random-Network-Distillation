import numpy as np
from model import Model


class Agent:
    def __init__(self, env, n_actions, n_states):

        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 5e-5
        self.n_actions = n_actions
        self.n_states = n_states
        self.max_steps = 100000
        self.env = env
        self.lr = 0.005
        self.target_model = Model(self.n_states, n_actions, self.lr, do_compile=False)
        self.eval_model = Model(self.n_states, n_actions, self.lr, do_compile=True)


    def choose_action(self, step, state):

        exp = np.random.rand()
        exp_probability = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-self.decay_rate * step)

        if exp < exp_probability:
            print(f"epsilon:{exp:0.3f}")
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.eval_model.predict(state))

    def train(self):
        state = self.env.reset()

        for step in range(self.max_steps):
            action = self.choose_action(step, state)
            next_state, reward, done, _, = self.env.step(action)
            self.env.render()
            if done:
                self.env.reset()
            print(f"step:{step}")
            state = next_state
