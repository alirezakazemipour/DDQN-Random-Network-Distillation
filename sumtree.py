import numpy as np



class SumTree:
    data_pointer = 0
    memory_counter = 0

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

    def update(self, index, p):

        delta_p = p - self.tree[index]
        self.tree[index] = p

        self.propagate(index, delta_p)

    def add(self, p, data):
        idx = self.data_pointer + self.transition_size -1

        self.data[self.data_pointer] = data
        self.update(idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.transition_size:
            self.data_pointer = 0

        if self.memory_counter < self.transition_size:
            self.memory_counter += 1

    def get_leaf_index(self, parent_idx, s):

        left_child = 2 * parent_idx + 1
        right_child = parent_idx + 1

        if left_child >= len(self.tree):
            return parent_idx

        if s <= self.tree[left_child]:
            return self.get_leaf_index(left_child, s)
        else:
            s -= self.tree[left_child]
            return self.get_leaf_index(right_child, s)

    def get(self, s):
        idx = self.get_leaf_index(0, s)
        p = self.tree[idx]
        data = self.data[idx - self.transition_size + 1]

        return idx, p, data

    def min_priority(self):
        return np.min(self.tree[self.transition_size -1:])