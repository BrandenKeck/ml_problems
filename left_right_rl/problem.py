import numpy as np

class LeftRight():

    def __init__(self, start, target, begin, end):
        self.states = np.arange(begin, end+1)
        self.current = start
        self.target = target
        self.max_idx = end - begin

    def get_idx(self):
        return np.where(self.states==self.current)[0][0]
    
    def move_left(self):
        idx = self.get_idx()
        if idx>0:
            self.current = self.states[idx-1]

    def move_right(self):
        idx = self.get_idx()
        if idx<self.max_idx:
            self.current = self.states[idx+1]

    def get_reward(self):
        return 1 - np.abs(self.target - self.current)
    
    def __repr__(self):
        return f'Position: {self.current}; Target: {self.target}'
            