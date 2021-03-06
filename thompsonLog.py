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

PRECISION = 300


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
		plt.ion()
		self.n = 0
		self.nQuestions = nQuestions
		self.results = []

		# thompson specific fit matrix cache
		self.axis = self.getAxis()
		self.axisN = len(self.axis)
		self.initParamCache()
		# self.f = plt.figure(1)
		# self.g = plt.figure(2)

	def getCurrSize(self):
		return (self.max + self.min) / 2

	def getNextSize(self):
		posterior = self.marginalize()
		# print(self.paramCache)
		# self.showScores()
		# self.showMarginal(posterior)
		# input()
		return self.thompsonChose(posterior)

	def recordResponse(self, size, correct):
		resultTuple = [size, 1 if correct else 0]
		# print(resultTuple)
		self.results.append(resultTuple)
		self.n += 1
		self.updateParamCache(resultTuple)

	def isDone(self):
		return self.n >= self.nQuestions

	def getAnswer(self):
		return self.getNextSize()

	def thompsonChose(self, posterior):
		axis = self.axis
		maxP = None
		argMax = None
		for i in range(len(axis)):
			value = axis[i]
			p = posterior[i]
			if maxP == None or p > maxP:
				maxP = p
				argMax = value
		return argMax

	# The param cache is the probability of the data given
	# particular values of k0 and k1. 
	# Rows correspond to k0
	# Cols correspond to k1
	def initParamCache(self):
		self.paramCache = np.ones((self.axisN, self.axisN))
		for i in range(self.axisN):
			for j in range(self.axisN):
				if i >= j:
					self.paramCache[i][j] = float('nan')

	def updateParamCache(self, result):
		for i in range(self.axisN):
			for j in range(self.axisN):
				if i < j: 
					self._updateHelper(result, i, j)

	def _updateHelper(self, result, i, j):

		# collect relevant data
		axis = self.axis
		k0 = axis[i]
		k1 = axis[j]
		x = result[0]
		y = result[1]

		# calculate p datum | parameters
		exponent = (x - k0) / (k1 - k0)
		p = 1 - math.pow(0.2, exponent)
		p = max(FLOOR, p)

		if y == 1:
			pDatum = p
		if y == 0:
			pDatum = (1- p)

		self.paramCache[i][j] *= pDatum

	def getAxis(self):
		return np.logspace(0.0, 1.0, num=PRECISION)

	def marginalize(self):
		axis = self.axis
		values = []
		for j in range(len(axis)):
			k1 = axis[j]
			marginalSum = 0
			n = 0
			for i in range(len(axis)):
				k0 = axis[i]
				if k0 >= k1: continue
				p = self.paramCache[i][j]
				marginalSum += p
				n +=1
			if n == 0:
				values.append(0)
			else:
				prior = self.priorK1(k1)
				values.append(prior * marginalSum / n)
		return values

	def priorK1(self, k1):
		return stats.norm.pdf(k1, loc=6.2, scale=3.5)

	def showMarginal(self, marginal):
		marginal = marginal / np.sum(marginal)
		axis = self.axis
		self.f.clear()
		plt.plot(axis, marginal)
		self.f.show()

	def showScores(self):
		self.g.clear()
		plt.imshow(self.paramCache, cmap=cm.coolwarm);
		plt.colorbar()
		ax = plt.gca();

		axis = self.axis

		axisLocations = []
		axisLabels = []
		for i in range(len(axis)):
			if i % 10 == 0:
				axisLocations.append(i)
				l = "{:.1f}".format(axis[i])
				axisLabels.append(l)
		ax.set_xticks(axisLocations)
		ax.set_xticklabels(axisLabels);
		ax.set_yticks(axisLocations)
		ax.set_yticklabels(axisLabels);
		self.g.show()

	