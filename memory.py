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

    def sample(self):
        pass

    def add(self, error, transition):
        p = self.get_priority(error)
        self.tree.add(p, transition)

    def get_priority(self, error):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        return p

    def update_tree(self, idx, p):
        self.tree.update(idx, p)
