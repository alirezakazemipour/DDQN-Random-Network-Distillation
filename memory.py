import numpy as np

class Memory:

    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_growth_rate = 0.001

    def __init__(self):
        pass

    def sample(self):
        pass

    def add(self):
        pass

    def get_priority(self, error):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        return p

    def update_tree(self):
        pass
