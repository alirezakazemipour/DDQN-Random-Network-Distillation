import numpy as np



class SumTree:
    def __init__(self, transition_size):

        self.transition_size = transition_size
        self.tree = np.zeros(2 * self.transition_size -1)
        self.data = np.zeros(self.transition_size, dtype=object)

    def total(self):
        return self.tree[0]

    def propagate(self, idx, delta_p):

        parent_idx = (idx - 1) // 2
        self.tree[parent_idx] += delta_p

        if parent_idx != 0:
            self.propagate(parent_idx, delta_p)

    def update(self):
        pass