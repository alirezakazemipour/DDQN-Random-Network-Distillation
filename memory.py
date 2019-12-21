import numpy as np
from sumtree import SumTree


class Memory:

    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_growth_rate = 0.001

    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.tree = SumTree(mem_size)

    def sample(self, k):

        batch = []
        segment = self.tree.total() // k
        indices = []
        IS = np.zeros(k)

        self.beta = np.min([1.0, self.beta + self.beta_growth_rate])

        for i in range(k):
            a = segment * i
            b = segment * (i+1)

            s = np.random.uniform(low = a, high = b)
            idx, priority, data = self.tree.get(s)

            IS[i] = np.power((self.tree.min_priority() / priority), self.beta)
            indices.append(idx)
            batch.append(data)

        return np.array(batch), indices, IS

    def add(self, error, transition):
        p = self.get_priority(error)
        self.tree.add(p, transition)

    def get_priority(self, error):
        p = np.clip(np.abs(error), -1, 1) # abs is unnecessary here ?
        p = (p + self.epsilon) ** self.alpha
        return p

    def update_tree(self, idx, p):
        self.tree.update(idx, p)
