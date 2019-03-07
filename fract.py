import numpy as np
import math
import random
import numpy.random as ra
import scipy.stats as stats
import tensorflow as tf

SEARCH_P = 0.5

# for the exponential fit
MIN_P = 1e-10
MAX_P = 0.9999999
FLOOR = (1. / 4.)
C = SEARCH_P
LEARN_F = -1

TRAIN_ITERS = 2000
ALPHA = 1e-3

# constants which is (1-FLOOR) /(SEARCH - FLOOR) - 1
K = (1 - FLOOR)/(SEARCH_P - FLOOR) - 1

# From the fract paper
# https://www.ipexhealth.com/wp-content/uploads/2018/01/FrACT-Landolt-Vision.pdf
# p(x) = f + (1 - f) * [1/(1+ [v0*x]^S)]
class Fract:

	def setFloorProbability(self, newFloor):
		global FLOOR
		FLOOR = newFloor

	@staticmethod
	def getParamValue(paramIndex):
		return 20 + 5 * paramIndex

	def __init__(self, nQuestions):
		self.n = 0
		self.nQuestions = nQuestions
		self.results = []

	def getNextSize(self):
		if self.n == 0:
			return 2
		nextSize = self.getAnswer()
		print('next size: {}'.format(nextSize))
		return nextSize

	def recordResponse(self, size, correct):
		resultTuple = [size, 1 if correct else 0]
		# print(resultTuple)
		self.results.append(resultTuple)
		self.n += 1

	def isDone(self):
		return self.n >= self.nQuestions

	def getAnswer(self):
		X,y = self.reformatData(self.results)
		fit = self.fractFit(X, y, FLOOR)
		[fitV0, fitS, fitF, loss] = fit
		xStar = math.pow(K, 1/fitS) / fitV0
		print(fitS, fitV0)
		return xStar
		# return self.getMostLikelyParticleK1()


	##############################

	def getBestNext(self):
		raise Exception('not used')
		return 2.0

	def fractFit(self, X, y, fixedF = LEARN_F):
		# make sure that the data is of the right form
		self.validateDataFormat(X, y)
		
		# variables
		label = tf.placeholder(tf.float32)
		x = tf.placeholder(tf.float32)

		# perhaps f is learned
		if fixedF != LEARN_F:
			f = tf.constant(fixedF, tf.float32)
		else:
			raise Exception('not used')
			f = tf.Variable([0.1],tf.float32)
		v0 = tf.Variable([2.2],  tf.float32)
		s  = tf.Variable([6.0] ,  tf.float32)

		# network
		# p(x) = f + (1 - f) * [1/(1+ [v0*x]^S)]
		p1 = 1.0/(1 + tf.pow(v0 * x, s))
		p2 = f + (1-f) * p1
		p3 = tf.clip_by_value(p2,MIN_P,MAX_P)
		networkLoss = tf.reduce_mean(-(label * tf.log(p3) + (1 - label) * tf.log(1 - p3)))

		# optimizer
		optimizer = tf.train.AdamOptimizer(ALPHA)
		trainer = optimizer.minimize(networkLoss)

		# session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)

		# before loss
		loss = sess.run(networkLoss, {x:X,label:y})
		
		for i in range(TRAIN_ITERS):
			loss = sess.run([trainer, networkLoss], feed_dict={x:X, label:y})
			# if i % (TRAIN_ITERS / 10) == 0:
			# 	print(loss[1])

		# after loss
		loss = sess.run(networkLoss, {x:X,label:y})

		# parameters
		fitF = sess.run(f)[0] if (fixedF == LEARN_F) else fixedF
		fitV0 = sess.run(v0)[0]
		fitS = sess.run(s)[0]
		return [fitV0, fitS, fitF, loss]

	def validateDataFormat(self, X, y):
		assert np.shape(X) == np.shape(y)
		for v in y:
			# all y values must be in {0, 1}
			assert v == 0.0 or v == 1.0

	def reformatData(self, data):
		n = len(data)
		X = np.zeros(n)
		y = np.zeros(n)
		for i in range(n):
			X[i] = data[i][0]
			y[i] = data[i][1]
		return X, y


	