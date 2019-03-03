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
DEFAULT_FLOOR = (1. / 4.)
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
		return 10 + 2 * paramIndex

	def __init__(self, maxN):
		plt.ion()
		self.slipP = 0.
		self.floorP = DEFAULT_FLOOR
		self.n = 0
		self.minN = 20
		self.maxN = 20
		self.minExpectedLoss = 0.001
		self.results = []

		# thompson specific fit matrix cache
		self.axis = self.getAxis()
		self.axisN = len(self.axis)
		self.initParamCache()
		self.f = plt.figure(1)
		# self.g = plt.figure(2)

	def setSlipProbability(self, slipP):
		self.slipP = slipP

	def setFloorProbability(self, floorP):
		self.floorP = floorP

	def getCurrSize(self):
		return (self.max + self.min) / 2

	def getNextSize(self):
		posterior = self.marginalize()

		# print(self.paramCache)
		# self.showScores()
		self.showMarginal(posterior)
		input()

		self.expectedLoss, self.argMin = self.getMinLoss()

		# return self.thompsonChose(posterior)
		return self.argMin

	def recordResponse(self, size, correct):
		resultTuple = [size, 1 if correct else 0]
		# print(resultTuple)
		self.results.append(resultTuple)
		self.n += 1
		self.updateParamCache(resultTuple)

	def isDone(self):

		if self.n < self.minN:
			return False
		if self.n >= self.maxN:
			return True
		
		return self.expectedLoss <= self.minExpectedLoss

	def getAnswer(self):
		return self.argMin

	def getMinLoss(self):
		posterior = self.marginalize()
		posterior = posterior / sum(posterior)
		minLoss = None
		argMinLoss = None
		for x_hat in self.axis:
			exLoss = 0
			for i in range(self.axisN):
				x_star = self.axis[i]
				p_x = posterior[i]
				exLoss += self.loss(x_hat, x_star) * p_x
			if minLoss == None or exLoss < minLoss:
				minLoss = exLoss
				argMinLoss = x_hat
		return minLoss, argMinLoss

	def loss(self, x_hat, x_star):
		return abs(x_star - x_hat)/x_star

	def thompsonChose(self, posterior):
		maxP = None
		argMax = None
		for i in range(self.axisN):
			value = self.axis[i]
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
		k0 = self.axis[i]
		k1 = self.axis[j]
		x = result[0]
		y = result[1]

		# calculate p datum | parameters
		exponent = (x - k0) / (k1 - k0)
		p = 1 - math.pow((1-C), exponent)
		p = max(self.floorP, p)

		# add in the slip
		p = self.slipP * self.floorP + (1 - self.slipP) * p

		if y == 1:
			pDatum = p
		if y == 0:
			pDatum = (1- p)

		self.paramCache[i][j] *= pDatum

	def getAxis(self):
		return np.arange(1.0, 10.0, 0.05)

	def marginalize(self):
		values = []
		for j in range(self.axisN):
			k1 = self.axis[j]
			marginalSum = 0
			n = 0
			for i in range(self.axisN):
				k0 = self.axis[i]
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
		# self.f.clear()
		plt.plot(self.axis, marginal)
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

	