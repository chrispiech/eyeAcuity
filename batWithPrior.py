import numpy as np
import math
import random
import numpy.random as ra
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

SLIP_P = 0.05

# model params
N_SAMPLES = 2000
MIN_START = 1
MAX_START = 10
MIN_N = 3

'''
Model:
Floored exponential acuity response curve
Model slip probability
Gumbel prior distribution on vision

Policy:
Likelihood weighting to sample from the posterior of P(theta | evidence)
Thomspon sampling to chose the next value of K1
'''
class BATPrior:

	def setFloorProbability(self, newFloor):
		global FLOOR
		FLOOR = newFloor

	@staticmethod
	def getParamValue(paramIndex):
		return 20 + 5 * paramIndex

	def __init__(self, nQuestions, prior):
		self.prior = prior
		self.n = 0
		self.nQuestions = nQuestions
		self.results = []

		# thompson specific fit matrix cache
		self.particles = self.getParamSamples()
		self.initWeights()
		# self.f = plt.figure(1)
		# self.g = plt.figure(2)

	def getCurrSize(self):
		raise Exception('not used')

	def getNextSize(self):
		return self.thompsonChose()

	def recordResponse(self, size, correct):
		resultTuple = [size, 1 if correct else 0]
		# print(resultTuple)
		self.results.append(resultTuple)
		self.n += 1
		self.updateWeights(resultTuple)

	def isDone(self):
		return self.n >= self.nQuestions

	def getAnswer(self):
		return self.getMostLikelyParticleK1()

	def getMostLikelyParticleK1(self):
		maxP = None
		argMax = None
		for particle in self.particles:
			# prior = self.getPriorP(particle)
			likelihood = particle['weight']
			p = likelihood
			if maxP == None or p > maxP:
				maxP = p
				argMax = particle['k1']
		return argMax

	def thompsonChose(self):
		sParticles = sorted(self.particles, key = lambda i: i['k1']) 
		cumProb = []
		cdf = 0
		for p in sParticles:
			normWeight = p['weight']
			cdf += normWeight
			cumProb.append(cdf)
		r = ra.uniform(0, 1)
		i = self.findIndex(r, cumProb)
		return sParticles[i]['k1']

 	# binary search
	def findIndex(self, goal, cdf):
		minI = 0
		maxI = len(cdf) - 1
		while True:
			currI = minI + (maxI - minI) // 2
			currV = cdf[currI]
			if maxI - minI <= 1:
				return maxI
			if currV < goal:
				minI = currI
			if currV >= goal:
				if cdf[currI-1] <= goal:
					return currI
				else:
					maxI = currI

	# The param weights are the probability of the data given
	# particular values of particle k0 and k1. 
	def initWeights(self):
		for particle in self.particles:
			particle['weight'] = 1.0 / len(self.particles)

	def updateWeights(self, result):
		weightSum = 0
		for particle in self.particles:
			self._updateHelper(result, particle)
			weightSum += particle['weight']
		for particle in self.particles:
			particle['weight'] /= weightSum

	def _updateHelper(self, result, particle):

		# collect relevant data
		k0 = particle['k0']
		k1 = particle['k1']
		x = result[0]
		y = result[1]

		# calculate p datum | parameters
		exponent = (x - k0) / (k1 - k0)
		try:
			p = 1 - math.pow(1 - C, exponent)
		# catch underflow errors
		except:
			p = FLOOR
		p = max(FLOOR, p)
		assert p < 1.000001

		if y == 1:
			pDatum = p
		if y == 0:
			pDatum = (1- p)

		particle['weight'] *= pDatum

	def getPriorP(self, particle):
		raise Exception('not working')
		k0 = particle['k0']
		k1 = particle['k1']

		logK0 = math.log(k0)
		logK1 = math.log(k1)

		muK0 = math.log(0.7 * k1, 10)
		pK1 = stats.gumbel_r.pdf(logK1, self.prior, 0.1)
		pK0 = stats.gumbel_r.pdf(logK0, muK0, 0.05)
		return pK0 * pK1

	def getParamSamples(self):
		samples = []
		for i in range(N_SAMPLES):
			muK1 = math.log(self.prior, 10)
			logK1 = self.rejectSampleGumbel(muK1, 0.1, -.3, 1.0)
			k1 = math.pow(10, logK1)
			muK0 = math.log(0.7 * k1, 10)
			logK0 = self.rejectSampleGumbel(muK0, 0.05, -10.0, logK1)
			k0 = math.pow(10, logK0)
			samples.append({
				'k0':k0,
				'k1':k1
			})
		return samples

	def rejectSampleGumbel(self,loc, scale, minV, maxV):
		while True:
			x = stats.gumbel_r.rvs(loc, scale)
			if x > minV and x < maxV:
				return x

	def marginalize(self):
		values = []
		for j in range(self.axisN):
			k1 = self.axis[j]
			marginalSum = 0
			n = 0
			for i in range(self.axisN):
				k0 = self.axis[i]
				if k0 >= k1: continue
				p = self.weightCache[i][j]
				marginalSum += p
				n +=1
			if n == 0:
				values.append(0)
			else:
				prior = self.priorK1(k1)
				values.append(prior * marginalSum / n)
		return values

	def showMarginal(self, marginal):
		marginal = marginal / np.sum(marginal)
		axis = self.axis()
		self.f.clear()
		plt.plot(axis, marginal)
		self.f.show()

	def showScores(self):
		self.g.clear()
		plt.imshow(self.weightCache, cmap=cm.coolwarm);
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

	