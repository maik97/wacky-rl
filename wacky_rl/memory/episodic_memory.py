from wacky_rl.memory import BasicMemory
import random

class EpisodicReplayBuffer:

    def __init__(self):
        self.clear()

    def clear(self):
        self.episodic_memory = []
        self.temp_memory = BasicMemory()

    def remember(self, list_of_items):
        self.temp_memory.remember(list_of_items)

    def recall_episodes(self, shuffle=False):
        if shuffle:
            random.shuffle(self.episodic_memory)
        mem_list = self.episodic_memory
        self.clear()
        return mem_list

    def finish_episode(self):
        self.episodic_memory.append(self.temp_memory.recall())