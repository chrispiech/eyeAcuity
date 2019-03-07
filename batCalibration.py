import numpy as np
import math
import random
import numpy.random as ra
import scipy.stats as stats
import matplotlib.pyplot as plt
from bayesianAcuityTest import *
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm

# some model constants for n guesses
MINIMUM_N = 10
MAXIMUM_N = 200 # nobody could possibly do a longer test
N_QUESTIONS = 20
# CONFIDENCE_GOAL = 0.95

'''
An extension of BayesianAcuityTest where
the testee requests a certain level of
precision. We only stop the test when we
have CONFIDENCE_GOAL probability that the test
error is less than the requested value
'''
class BATCalibration(BayesianAcuityTest):

	'''
	Precision scores are relative error eg:
	0.1 is fine, 0.05 is quite good, 0.02 is
	impeccible.
	'''

	def __init__(self, precisionRange):
		self.n = 0
		self.nQuestions = N_QUESTIONS
		self.results = []

		self.precisionRange = precisionRange

		# thompson specific fit matrix cache
		self.particles = self.getParamSamples()
		self.initWeights()


	# for all possible answers, calculate the
	# probability that the correct answer is within
	# the error bounds of the supposed answer. Return
	# both the best answer and the confidence
	def getBestAnswerSoFar(self):
		cdf = self.calcCdf()
		delta = 0.01
		logMars = np.arange(MIN_LOG_MAR, MAX_LOG_MAR + delta, delta)
		argMax = None
		maxProb = None
		for lm in logMars:
			xHat = math.pow(10, lm)
			p = self.calcLikelihoodInRange(cdf, xHat)
			if argMax == None or p > maxProb:
				maxProb = p
				argMax = xHat
		return argMax, maxProb

	def calcLikelihoodInRange(self, cdf, xHat):
		maxX = xHat * (1 + self.precisionRange)
		minX = xHat * (1 - self.precisionRange)
		yLarge = self.estimateCdf(cdf, maxX)
		ySmall = self.estimateCdf(cdf, minX)
		return yLarge - ySmall

	# A little complicated. I read through it thuroughly
	# but did not test it explicitly
	def estimateCdf(self, cdf, x):
		minI = 0
		maxI = len(cdf) - 1
		# edge case to handle: x < min(cdf) or x > max(cdf)
		if x < cdf[0]['x']: return 0.
		if x >= cdf[len(cdf)-1]['x']: return 1.
		'''
		Idea: Binary search for the index just above x
		'''
		while True:
			# you have found your interval
			if maxI - minI <= 1:
				return self.getCdfY(cdf, maxI, x)

			currI = minI + (maxI - minI) // 2
			currX = cdf[currI]['x']
			# print(currX)
			if currX < x:
				minI = currI
			if currX >= x:
				maxI = currI


	def getCdfY(self, cdf, upperIndex, x):
		upper = cdf[upperIndex]
		lower = cdf[upperIndex - 1]

		# linearly interpolate
		rise = upper['cdf(x)'] - lower['cdf(x)']
		run = upper['x'] - lower['x']

		# is the fraction of the "run" that x has done
		t = (x - lower['x']) / run

		return lower['cdf(x)'] + t * rise

	def calcCdf(self):
		sParticles = sorted(self.particles, key = lambda i: i['k1'])
		cumulative = 0
		cdf = []
		for p in sParticles:
			x = p['k1']
			normWeight = p['weight']
			cumulative += normWeight
			cdf.append({
				'x':x,
				'cdf(x)':cumulative
			})
		return cdf

