from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.num_experiences = 0
        self.buffer = deque()
		
    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.num_experiences

    def add(self, state, action, reward, new_state):
        experience = (state, action, reward, new_state)
        if self.num_experiences < self.max_memory:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.forget()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

	def forget(self):
		self.buffer.popleft()
	
    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0