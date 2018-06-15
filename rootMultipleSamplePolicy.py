SEARCH_P = 0.8
INIT_X = 9
A_IN = 14


from collections import defaultdict, Counter

class RootMultipleSamplePolicy:

	@staticmethod
	def getParamValue(paramIndex) :
		# param is number of iterations
		return 10 + 2 * paramIndex

	def __init__(self, nIterations):
		self.nIterations = nIterations
		self.nTestsPerSize = 5

		self.theta = INIT_X
		self.rootN = 1

		self.n = 0

		self.nAtSize = 0
		self.nCorrectAtSize = 0


	def getAvgY(self, size, wasCorrect):
		# return 1
		return self.nCorrectAtSize / self.nAtSize


	def recordResponse(self, size, wasCorrect):
		def a(n):
			return A_IN/n

		if wasCorrect:
			self.nCorrectAtSize += 1


		self.nAtSize += 1
		self.n += 1




		if self.nAtSize == self.nTestsPerSize:

			y = SEARCH_P

			# avg_y = 1
			avg_y = self.getAvgY(size, wasCorrect)
			y_n = avg_y


			self.theta = self.theta - a(self.rootN) * (y_n - y)

			self.nAtSize = 0
			self.nCorrectAtSize = 0

			self.rootN += 1





	def getNextSize(self):
		return self.theta

	def isDone(self):
		return self.n > self.nIterations

	def getAnswer(self):
		return self.theta