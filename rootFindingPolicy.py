class RootFindingPolicy:

	@staticmethod
	def getParamValue(paramIndex) :
		return 1

	def __init__(self, paramValue):
		self.nTested = 0

	def getNextSize(self):
		return 5

	def recordResponse(self, size, ans):
		self.nTested += 1

	def isDone(self):
		return self.nTested > 1

	def getAnswer(self):
		return self.paramValue