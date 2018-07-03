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

class ThompsonPolicy:


	@staticmethod
	def getParamValue(paramIndex):
		return 20 + 5 * paramIndex

	def __init__(self, nQuestions):
		self.n = 0
		self.nQuestions = nQuestions
		self.results = []

	def getCurrSize(self):
		return (self.max + self.min) / 2

	def getNextSize(self):
		fitMatrix = self.calcFitMatrix()
		posterior = self.marginalize(fitMatrix)
		# posterior = posterior / np.sum(posterior)
		return self.thompsonChose(posterior)

	def recordResponse(self, size, correct):
		resultTuple = [size, 1 if correct else 0]
		self.results.append(resultTuple)
		self.n += 1

	def isDone(self):
		return self.n >= self.nQuestions

	def getAnswer(self):
		return self.getNextSize()

	def thompsonChose(self, posterior):
		axis = self.getAxis()
		maxP = None
		argMax = None
		for i in range(len(axis)):
			value = axis[i]
			p = posterior[i]
			if maxP == None or p > maxP:
				maxP = p
				argMax = value
		# f = plt.figure(1)
		# plt.plot(axis, posterior)
		# f.show()
		# input()
		return argMax

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

	def calcFitMatrix(self):
		axis = self.getAxis()
		n = len(axis)
		X,y = self.reformatData()
		fitMatrix = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				k0 = axis[i]
				k1 = axis[j]
				if k0 >= k1: 
					fitMatrix[i][j] = float('nan')
				else:
					score = self.calcProbData(k0, k1, X, y)
					fitMatrix[i][j] = score
		return fitMatrix

	def getAxis(self):
		return np.arange(1.0, 10.0, 0.1)

	def exploreFit(self):
		fitMatrix = self.calcFitMatrix()
		posterior = self.marginalize(fitMatrix)
		raise Exception('todo')

	def marginalize(self, fitMatrix):
		axis = self.getAxis()
		values = []
		for j in range(len(axis)):
			k1 = axis[j]
			marginalSum = 0
			n = 0
			for i in range(len(axis)):
				k0 = axis[i]
				if k0 >= k1: continue
				p = fitMatrix[i][j]
				marginalSum += p
				n +=1
			if n == 0:
				values.append(MIN_P)
			else:
				prior = self.priorK1(k1)
				values.append(prior * marginalSum / n)
		return values

	def priorK1(self, k1):
		return stats.norm.pdf(k1, loc=6.2, scale=3.5)

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

	