from collections import deque
from random import sample

class Record:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class ReplayMemory:
    def __init__(self, capacity: int = 10000):
        self.memory: deque[Record] = deque([], maxlen=capacity)

    def push(self, record: Record):
        self.memory.append(record)

    def sample(self, batch_size = 128):
        return sample(self.memory, batch_size)