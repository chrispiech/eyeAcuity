import tensorflow as tf
import numpy as np
import math
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm

SEARCH_P = 0.8
INIT_X = 9
A_IN = 10

# for the exponential fit
MIN_P = 1e-10
MAX_P = 0.9999999
FLOOR = (1. / 4.)
C = 0.8
INF = 9999999


# beta

MIN_START = 1
MAX_START = 10
MIN_N = 3

#this is the one where I use the OG formula

class ExponentialFitPolicy:


	@staticmethod
	def getParamValue(paramIndex):
		# params are thresholds
		# range from .7 to .99
		
		return min(0.92, 0.85 + (paramIndex * 0.02))
		#return 2 + paramIndex

	def __init__(self, threshold):
		self.threshold = threshold
		self.maxDepth = 4

		self.min = MIN_START
		self.max = MAX_START
		self.depth = 0

		self.results = []

		self.resetBeta()

	def getCurrSize(self):
		return (self.max + self.min) / 2

	def getNextSize(self):
		return self.getCurrSize()

	def recordResponse(self, size, correct):
		resultTuple = [size, 1 if correct else 0]
		self.results.append(resultTuple)

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
		return self.exploreFit()

	def resetBeta(self):
		self.n = 0
		self.a = 0.5
		self.b = 0.5

	def lambdaFromK1(self, b_start):
		k1 = self.getCurrSize()
		return -np.log(0.2)/(k1 - b_start)

	# note that this is proportional to the
	# true probability
	def calcProbData(self, k0, k1, X, y):
		n = len(X)
		logP = 0
		for i in range(n):
			exponent = (X[i] - k0) / (k1 - k0)
			p = 1 - math.pow(0.2, exponent)
			p = max(FLOOR, p)
			p = min(MAX_P, p)

			bce = (y[i] * np.log(p) + (1. - y[i]) * np.log(1. - p))
			logP += bce
		return np.exp(logP)

	def exploreFit(self):
		X,y = self.reformatData()

		k0axis = np.arange(1.0, 10.0, 0.1)
		k1axis = np.arange(1.0, 10.0, 0.1)
		n = len(k0axis)

		fitMatrix = np.zeros((n, n))
		maxScore = None
		maxParams = None

		print('X: ', X)
		print('y: ', y)


		for i in range(n):
			for j in range(n):
				k0 = k0axis[i]
				k1 = k1axis[j]
				if k0 >= k1: 
					fitMatrix[i][j] = float('nan')
				else:
					score = self.calcProbData(k0, k1, X, y)
					fitMatrix[i][j] = score
					if maxScore == None or score > maxScore:
						maxScore = score
						maxParams = (k0, k1)

		print('score: ', maxScore)
		print('params:  ', maxParams)

		values = self.marginalizek0(fitMatrix, k0axis)

		self.showScores(fitMatrix, k0axis)
		self.showMargninal(values, k0axis)

		input()

		raise Exception('todo')

	def showMargninal(self, values, k0axis):
		g = plt.figure(2)
		plt.plot(k0axis, values)
		g.show()

	def marginalizek0(self, fitMatrix, k0axis):
		values = []
		argMax = None
		maxValue = None
		for j in range(len(k0axis)):
			k1 = k0axis[j]
			marginalSum = 0
			n = 0
			for i in range(len(k0axis)):
				k0 = k0axis[i]
				if k0 >= k1: continue
				p = fitMatrix[i][j]
				marginalSum += p
				if maxValue == None or p > maxValue:
					maxValue = p
					argMax = k1
				n +=1
			if n == 0:
				values.append(0)
			else:
				prior = self.priorK1(k1)
				values.append(prior * marginalSum / n)
		print('argmax: ', argMax)
		return values

	def priorK1(self, k1):
		return 1
		#return stats.norm.pdf(k1, loc=6.2, scale=3)

	def showScores(self, fitMatrix, k0axis):
		f = plt.figure(1)
		plt.imshow(fitMatrix, cmap=cm.coolwarm);
		plt.colorbar()
		ax = plt.gca();

		axisLocations = []
		axisLabels = []
		for i in range(len(k0axis)):
			if i % 10 == 0:
				axisLocations.append(i)
				l = "{:.1f}".format(k0axis[i])
				axisLabels.append(l)
		ax.set_xticks(axisLocations)
		ax.set_xticklabels(axisLabels);
		ax.set_yticks(axisLocations)
		ax.set_yticklabels(axisLabels);
		f.show()

	def calcInv(self, b, lam):
		return b - (1. / lam) * math.log(1. - C)

	def reformatData(self):
		n = len(self.results)
		X = np.zeros(n)
		y = np.zeros(n)
		for i in range(n):
			# now x is in terms of "difficulty"
			X[i] = self.results[i][0]
			y[i] = self.results[i][1]
		return X, y

	