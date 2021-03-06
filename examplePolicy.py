class ExamplePolicy:

	@staticmethod
	def getParamValue(paramIndex) :
		return 1 + paramIndex

	def __init__(self, paramValue):
		self.paramValue = paramValue
		self.nTested = 0

	def getNextSize(self):
		return 5

	def recordResponse(self, size, ans):
		self.nTested += 1

	def isDone(self):
		return self.nTested > 1

	def getAnswer(self):
		return self.paramValue