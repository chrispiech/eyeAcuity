import scipy.stats as stats

SEARCH_P = 0.8

MIN_START = 1
MAX_START = 10
MIN_N = 3

class BinaryBetaPolicy:

	@staticmethod
	def getParamValue(paramIndex):
		# params are thresholds
		# range from .7 to .99
		
		return min(0.99, 0.76 + (paramIndex * 0.02))
		#return 2 + paramIndex

	def __init__(self, threshold):
		self.threshold = threshold
		self.maxDepth = 4

		self.min = MIN_START
		self.max = MAX_START
		self.depth = 0

		self.resetBeta()

	def getCurrSize(self):
		return (self.max + self.min) / 2

	def getNextSize(self):
		return self.getCurrSize()

	def recordResponse(self, size, correct):
		self.n += 1
		if correct:
			self.a += 1
		else:
			self.b += 1
		beta = stats.beta(self.a, self.b)
		pUnderSearch = beta.cdf(SEARCH_P)
		pOverSearch = 1.0 - pUnderSearch
		#print(self.getCurrSize(), self.n, self.a, self.b, pUnderSearch, pOverSearch)
		if self.n >= MIN_N and pOverSearch >= self.threshold:
			#print('too large...')
			self.max = self.getCurrSize()
			self.depth += 1
			self.resetBeta()
			#print('')
		if self.n >= MIN_N and pUnderSearch >= self.threshold:
			#print('too small...')
			self.min = self.getCurrSize()
			self.depth += 1
			self.resetBeta()
			#print('')

	def isDone(self):
		return self.depth >= self.maxDepth

	def getAnswer(self):
		return self.getCurrSize()

	def resetBeta(self):
		self.n = 0
		self.a = 0.5
		self.b = 0.5