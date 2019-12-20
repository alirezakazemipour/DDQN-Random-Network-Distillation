import numpy as np
from model import Model
from memory import Memory


class Agent:
    def __init__(self, env, n_actions, n_states):

        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 5e-5
        self.n_actions = n_actions
        self.n_states = n_states
        self.max_steps = 100000
        self.max_episodes = 500
        self.mem_size = 0.8 * self.max_steps
        self.env = env
        self.recording_counter = 0
        self.batch_size = 32
        self.lr = 0.005
        self.gamma = 0.99
        self.target_model = Model(self.n_states, n_actions, self.lr, do_compile=False)
        self.eval_model = Model(self.n_states, n_actions, self.lr, do_compile=True)
        self.memory = Memory(self.mem_size)

    def choose_action(self, step, state):

        exp = np.random.rand()
        exp_probability = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-self.decay_rate * step)

        if exp < exp_probability:
            print("epsilon:{:0.3f}".format(exp))
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.eval_model.predict(state))

    def update_train_model(self):
        self.target_model.set_weights(self.eval_model.get_weights())

    def train(self):

        batch, indices, IS = self.memory.sample(np.min[self.batch_size, self.recording_counter])


    def append_transition(self, transition):

        self.recording_counter += 1

        next_state, reward, action, state, done = zip(*list(transition))

        y = np.max(self.target_model.predict([next_state]))
        y = reward + self.gamma * y * (1 - done)

        y_train = self.eval_model.predict([state])

        y_train[np.arange(self.batch_size), action] = y

        td_error = np.abs(y - y_train)

        self.memory.add(td_error, transition) #Unzip where you need

    def run(self):

        for episodes in range(self.max_episodes):
            state = self.env.reset()

            for step in range(self.max_steps):

                action = self.choose_action(step, state)
                next_state, reward, done, _, = self.env.step(action)
                transition = zip(next_state, reward, action, state, done)
                self.append_transition(transition)

                self.env.render()
                if done:
                    self.env.reset()
                print("step:{}".format(step))
                state = next_state
            self.train()
