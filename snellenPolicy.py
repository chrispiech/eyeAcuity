SIZES = [
	10, 
	5,
	3.5,
	2.5,
	2.0,
	1.5,
	1.25,
	1
]

SEARCH_P = 0.8

class ManualPolicy:

	@staticmethod
	def getParamValue(paramIndex) :
		# The param is the number of tests at size
		return 5 + paramIndex

	def __init__(self, nTestsPerSize):
		self.nTestsPerSize = nTestsPerSize

		self.nAtSize = 0
		self.nCorrectAtSize = 0

		self.sizeIndex = 0
		self.done = False
		self.finalAnswer = None

	def getNextSize(self):
		return SIZES[self.sizeIndex]

	def recordResponse(self, size, isCorrect):
		if isCorrect:
			self.nCorrectAtSize += 1
		self.nAtSize += 1
		if self.nAtSize == self.nTestsPerSize:
			self.changeSize()

	def isDone(self):
		return self.done

	def getAnswer(self):
		return self.finalAnswer

	def changeSize(self):
		pCorrect = float(self.nCorrectAtSize) / self.nTestsPerSize
		if pCorrect <= SEARCH_P:
			self.finalAnswer = SIZES[self.sizeIndex]
			self.done = True
			return

		self.sizeIndex += 1
		self.nAtSize = 0
		self.nCorrectAtSize = 0

		if self.sizeIndex == len(SIZES):
			self.finalAnswer = SIZES[len(SIZES) - 1]
			self.done = True
			return

