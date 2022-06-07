from wacky import functional as funky

class BaseMemory(funky.WackyBase):

	def __init__(self, max_len):
		super().__init__()

		self.max_len = max_len

	def call(self, key, idx=None):
		pass

