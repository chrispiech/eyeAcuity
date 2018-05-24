SEARCH_P = 0.8
INIT_X = 9
A_IN = 10

class RootFindingPolicy:

	@staticmethod
	def getParamValue(paramIndex) :
		# param is number of iterations
		return 10 + 2 * paramIndex

	def __init__(self, nIterations):
		self.nIterations = nIterations
		self.theta = INIT_X
		self.n = 1

	def recordResponse(self, size, wasCorrect):
		def a(n):
			return A_IN/n

		y = SEARCH_P
		y_n = 1 if wasCorrect else 0
		
		self.theta = self.theta - a(self.n) * (y_n - y)
		self.n += 1

	def getNextSize(self):
		return self.theta

	def isDone(self):
		return self.n > self.nIterations

	def getAnswer(self):
		return self.theta