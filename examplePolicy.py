class ExamplePolicy:


	def __init__(self, alpha):
		self.alpha = alpha
		self.nTested = 0

	def getNextSize(self):
		return 5

	def recordResponse(self, size, ans):
		self.nTested += 1

	def isDone(self):
		return self.nTested > 20

	def getAnswer(self):
		return 5