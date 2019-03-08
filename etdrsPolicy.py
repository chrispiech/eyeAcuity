# typically has 10 types of letters
# stopping condition: < 50% of letters are identified

'''
I implemented ETDRS as explained here:
http://www.vectorvision.com/clinical-use-etdrs-acuity/

Determine the log score for the last passed row
subtract 0.02 log units for every letter that is correctly 
identified beyond the last passed row. For example, if the 
patient reads all of the letters correctly on the 20/30 row 
and then 3 letters correctly on the 20/25 row, the Log Score 
would be calculated as follows:
20/30 Row = 0.20
3 letters X 0.02 log/letter = – 0.06
ETDRS Acuity Log Score = 0.14
'''

import math

# line sizes of the ETDRS chart,
# in 1/decimal notation
SIZES = [
	10,
	8,
	6.25,
	5,
	4,
	3.15,
	2.5,
	2,
	1.6,
	1.25,
	1,
	0.8,
	0.625,
	0.5,
]


class ETDRSPolicy:

	@staticmethod
	def getParamValue(paramIndex) :
		# The param is the number of tests at size
		return 5

	def __init__(self, nTestsPerSize):
		self.nTestsPerSize = 5

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
		start = self.finalAnswer
		logScore = math.log(start, 10)
		n = self.nCorrectAtSize
		logScore -= n * 0.02
		score = math.pow(10, logScore)
		return score

	def setFloorProbability(self, floorP):
		pass

	def changeSize(self):
		pCorrect = float(self.nCorrectAtSize) / self.nTestsPerSize
		if pCorrect <= 0.5:
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

