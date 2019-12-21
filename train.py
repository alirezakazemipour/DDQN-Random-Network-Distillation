import numpy as np
from model import model
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
        self.target_update_period = 15
        self.mem_size = int(0.8 * self.max_steps)
        self.env = env
        self.recording_counter = 0
        self.batch_size = 32
        self.lr = 0.005
        self.gamma = 0.99
        self.target_model = model(self.n_states, n_actions, self.lr, do_compile=False)
        self.eval_model = model(self.n_states, n_actions, self.lr, do_compile=True)
        self.memory = Memory(self.mem_size)

    def choose_action(self, step, state):

        exp = np.random.rand()
        exp_probability = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-self.decay_rate * step)

        if exp < exp_probability:
            # print("epsilon:{:0.3f}".format(exp))
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.eval_model.predict(state))

    def update_train_model(self):
        self.target_model.set_weights(self.eval_model.get_weights())

    def train(self):

        batch, indices, IS = self.memory.sample(np.min[self.batch_size, self.recording_counter])

        state = batch[:, self.n_states]
        reward = batch[:, self.n_states]
        action = batch[:, self.n_states + 1].astype("int")
        next_state = batch[:, self.n_states + 2:-1]
        done = batch[:, -1]
        # print("next_state:", next_state)

        # next_state = np.expand_dims( next_state, axis=0 )
        # print( "next_state shape:", next_state.shape )

        target_q = np.max( self.target_model.predict( next_state ) )
        target_q = reward + self.gamma * target_q * (1 - done)

        # state = np.expand_dims( state, axis=0 )
        eval_q = self.eval_model.predict( state )

        y = eval_q.copy()

        y[np.arange( self.batch_size ), action] = target_q

        self.eval_model.train_on_batch( state, np.concatenate( [y, IS], axis=-1 ) )

        self.memory.update_tree(indices, self.abs_error)

    def append_transition(self, transition):

        self.recording_counter += 1

        state = transition[:self.n_states]
        reward = transition[self.n_states]
        action = transition[self.n_states + 1].astype("int")
        next_state = transition[self.n_states +2:-1]
        done = transition[-1]
        # print("next_state:", next_state)

        next_state = np.expand_dims(next_state, axis = 0)
        print("next_state shape:", next_state.shape)

        target_q = np.max(self.target_model.predict(next_state))
        target_q = reward + self.gamma * target_q * (1 - done)

        state = np.expand_dims(state, axis = 0)
        eval_q = self.eval_model.predict(state)

        eval_q = np.squeeze(eval_q, axis = 0)

        y = eval_q.copy()

        print("y shape:{}".format(y.shape))

        y[np.arange(self.batch_size), action] = target_q

        self.abs_error = np.abs(y - eval_q)

        self.memory.add(self.abs_error, transition) #Unzip where you need

    def run(self):

        for episode in range(self.max_episodes):
            state = self.env.reset()

            for step in range(self.max_steps):

                action = self.choose_action(step, state)
                next_state, reward, done, _, = self.env.step(action)
                # print( "next_state:", next_state)
                transition = np.array(list(state) + [reward] + [action] + list(next_state) + [done])
                self.append_transition(transition)

                self.env.render()
                if done:
                    self.env.reset()
                print("step:{}".format(step))
                state = next_state
            self.train()

            if episode % self.target_update_period == 0:
                self.update_train_model()
