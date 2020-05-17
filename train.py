import numpy as np
from model import Model, RNDModel
from memory import Memory, Transition
import torch
from torch import device
from torch import from_numpy
from torch.optim import Adam


class Agent:
    def __init__(self, env, n_actions, n_states, n_encoded_features):

        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 5e-3
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_encoded_features = n_encoded_features
        self.max_steps = 500
        self.max_episodes = 500
        self.target_update_period = 300
        self.mem_size = 10000
        self.env = env
        self.recording_counter = 0
        self.batch_size = 128
        self.lr = 0.01
        self.gamma = 0.95
        self.device = device("cpu")

        self.q_target_model = Model(self.n_states, self.n_actions).to(self.device)
        self.q_eval_model = Model(self.n_states, self.n_actions).to(self.device)
        self.q_target_model.load_state_dict(self.q_eval_model.state_dict())

        self.rnd_predictor_model = RNDModel(self.n_states, self.n_encoded_features)
        self.rnd_target_model = RNDModel(self.n_states, self.n_encoded_features)

        self.memory = Memory(self.mem_size)

        self.loss_fn = torch.nn.MSELoss()
        self.q_optimizer = Adam(self.q_eval_model.parameters(), lr=self.lr)
        self.feature_optimizer = Adam(self.rnd_predictor_model.parameters(), lr=self.lr)

    def choose_action(self, step, state):

        exp = np.random.rand()
        exp_probability = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-self.decay_rate * step)

        if exp < exp_probability:
            return np.random.randint(self.n_actions), exp_probability
        else:
            state = np.expand_dims(state, axis=0)
            state = from_numpy(state).float().to(self.device)
            return np.argmax(self.q_eval_model(state).detach().numpy()), exp_probability

    def update_train_model(self):
        self.q_target_model.load_state_dict(self.q_eval_model.state_dict())

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0  # as no loss
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)

        x = states
        q_eval = self.q_eval_model(x).gather(dim=1, index=actions.long())
        i_rewards = self.get_intrinsic_reward(states.detach().numpy())
        with torch.no_grad():
            q_next = self.q_target_model(next_states)

            q_eval_next = self.q_eval_model(next_states)
            max_action = torch.argmax(q_eval_next, dim=-1)

            batch_indices = torch.arange(end=self.batch_size, dtype=torch.int32)
            target_value = q_next[batch_indices.long(), max_action] * (1 - dones)

            q_target = i_rewards.detach() + rewards + self.gamma * target_value
        loss = self.loss_fn(q_eval, q_target.view(self.batch_size, 1))
        predictor_loss = i_rewards.sum()

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        dqn_loss = loss.detach().cpu().numpy()

        self.feature_optimizer.zero_grad()
        predictor_loss.backward()
        self.feature_optimizer.step()
        rnd_loss = predictor_loss.detach().cpu().numpy()

        return dqn_loss, rnd_loss

    def run(self):

        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            for step in range(self.max_steps):
                action, random_action_prob = self.choose_action(episode, state)
                next_state, reward, done, _, = self.env.step(action)
                episode_reward += reward
                total_reward = reward + self.get_intrinsic_reward(np.expand_dims(state, 0)).detach().clamp(-1, 1)
                self.store(state, total_reward, done, action, next_state)
                dqn_loss, rnd_loss = self.train()
                if done:
                    break
                state = next_state

            if episode % self.target_update_period == 0:
                self.update_train_model()
            print(f"EP:{episode}| "
                  f"DQN loss:{dqn_loss}| "
                  f"RND loss:{rnd_loss}| "
                  f"EP_reward:{episode_reward}| "
                  f"Random action prob:{random_action_prob}| "
                  f"Step:{step}| "
                  f"Memory size:{len(self.memory)}")

    def store(self, state, reward, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        reward = torch.Tensor([reward]).to("cpu")
        done = torch.Tensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, reward, done, action, next_state)

    def get_intrinsic_reward(self, x):
        # x = np.expand_dims(x, axis=0)
        x = from_numpy(x).float().to(self.device)
        predicted_features = self.rnd_predictor_model(x)
        target_features = self.rnd_target_model(x).detach()

        intrinsic_reward = (predicted_features - target_features).pow(2).sum(1)
        return intrinsic_reward

    def unpack_batch(self, batch):

        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).to(self.device).view(self.batch_size, self.n_states)
        actions = torch.cat(batch.action).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device).view(self.batch_size, self.n_states)
        dones = torch.cat(batch.done).to(self.device)
        actions = actions.view((-1, 1))
        return states, actions, rewards, next_states, dones
