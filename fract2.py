import numpy as np
import math
import random
import numpy.random as ra
import scipy.stats as stats
import tensorflow as tf

SEARCH_P = 0.8

# for the exponential fit
MIN_P = 1e-10
MAX_P = 0.9999999
FLOOR = (1. / 4.)
C = SEARCH_P

# constants which is (1-FLOOR) /(SEARCH - FLOOR) - 1
K = (1 - FLOOR)/(SEARCH_P - FLOOR) - 1
N = 10000
MIN_LOG_MAR = -0.3
MAX_LOG_MAR = 1.0

# From the fract paper
# https://www.ipexhealth.com/wp-content/uploads/2018/01/FrACT-Landolt-Vision.pdf
# p(x) = f + (1 - f) * [1/(1+ [v0/x]^S)]
# As in the 2006 paper, I allow for S to be a free variable
# To make sure that I get the right answer, I generate 10k v0, s parameters
# I find the maximum likelihood params.. and just like in the paper when chosing
# a next size, I chose the v0 which maximizes likelihood (translated back into)
# standard units. When its time to provide an answer I chose the max likelihood
# v0 and s, and reverse calculate what font size has a probability of SEARCH_P

# Very Important: FrACT is defined in "decimal" units. Our experiment uses
# "standard units". standard = 1 / decimal
class Fract2:

	def setFloorProbability(self, newFloor):
		if(newFloor != FLOOR):
			raise Exception('not implemented')

	@staticmethod
	def getParamValue(paramIndex):
		return 20

	def __init__(self, nQuestions):
		self.n = 0
		self.nQuestions = nQuestions
		self.results = []
		self.samples, self.weights = self.makeSamples()

	def getNextSize(self):
		if self.n == 0:
			return 2 # this is the mode
		v0, s = self.getBestParticle()

		nextSize = 1/ v0
		# print('next size: {}'.format(nextSize))
		# convert back into standard units
		return nextSize

	def recordResponse(self, size, correct):
		resultTuple = [size, 1 if correct else 0]
		# note that results are in standard units
		self.results.append(resultTuple)
		self.n += 1

	def isDone(self):
		return self.n >= self.nQuestions

	# make sure to return standard units
	def getAnswer(self):
		v0, s = self.getBestParticle()
		# in decimal units
		x_star = v0 / math.pow(K, 1/s)
		# print('best particle', v0, s)
		return 1 / x_star
		
	# for this calculation, all terms should be
	# in decimal units
	def getBestParticle(self):
		lastResponse = self.results[-1]
		# size in decimal
		vi = 1/ lastResponse[0]
		yi = lastResponse[1]

		argMax = None
		valMax = None
		for i in range(N):
			v0, s = self.samples[i]
			self.weights[i] *= self.getLikelihood(v0, vi, s, yi)

			if argMax == None or self.weights[i] > valMax:
				valMax = self.weights[i]
				# we are going to return in standard units
				argMax = self.samples[i]
		return argMax

	def getLikelihood(self, v0, vi, s, yi):
		# term = (v0/vi)^s
		term = math.pow(v0/vi, s)
		# px is probability with no guessing
		px = 1 / (1 + term)
		# pi is the probability correct
		pi = FLOOR + (1-FLOOR)*px
		# bern likelihood
		if yi == 1: 
			return pi
		return 1 - pi

	# returns samples in decimal units!
	def makeSamples(self):
		samples = []
		weights = []
		for i in range(N):
			logMarV0 = np.random.uniform(MIN_LOG_MAR,MAX_LOG_MAR,1)
			logS = np.random.uniform(0.05, 2.2)

			# convert into "decimal" units
			v0_r = math.pow(10, logMarV0)
			v0 = 1/ v0_r

			# note that these are equal
			# logMarV0 == -math.log(v0, 10)

			s = -math.pow(10, logS)
			samples.append([v0, s])
			weights.append(1.0)
		return samples, weights


	