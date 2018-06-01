import tensorflow as tf
import numpy as np
import math
import random

SEARCH_P = 0.7
INIT_X = 9
A_IN = 10

# for the exponential fit
MIN_P = 1e-10
MAX_P = 0.9999999
FLOOR = (1. / 4.)
C = 0.8
INF = 9999999

#this is the one where I use the OG formula

class ExponentialFitPolicy:

	@staticmethod
	def getParamValue(paramIndex) :
		# param is number of iterations
		return 30 + 2 * paramIndex

	def __init__(self, nIterations):
		self.nIterations = nIterations
		self.theta = INIT_X
		self.n = 1
		self.results = []

	def recordResponse(self, size, wasCorrect):
		def a(n):
			return A_IN/n

		y = SEARCH_P
		y_n = 1 if wasCorrect else 0
		
		self.theta = self.theta - a(self.n) * (y_n - y)
		self.n += 1

		resultTuple = [size, y_n]
		self.results.append(resultTuple)

	def getNextSize(self):
		return self.theta

	def isDone(self):
		return self.n > self.nIterations

	def getAnswer(self):
		return self.exponentialFit()

	def exponentialFit(self):
		print('FIT')
		X,y = self.reformatData()
		
		#print(X,y)
		# variables
		label = tf.placeholder(tf.float32)
		x = tf.placeholder(tf.float32)
		b = tf.Variable([5.], tf.float32)
		lam = tf.Variable([1.], tf.float32)

		# network
		px = 1 - tf.exp(-lam * (x - b))
		p = tf.maximum(px, FLOOR)
		p = tf.clip_by_value(px,MIN_P,MAX_P)
		networkLoss = tf.reduce_mean(-(label * tf.log(p) + (1. - label) * tf.log(1. - p)))

		# optimizer
		optimizer = tf.train.AdamOptimizer(5e-4)
		trainer = optimizer.minimize(networkLoss)

		# session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)

		oldLoss = INF
		flag = False
		for i in range(100000):
			output = sess.run([trainer, networkLoss], feed_dict={x:X, label:y})
			loss = output[1]
			curr_b = sess.run(b)[0]
			curr_lam = sess.run(lam)[0]
			curr_k1 = self.calcInv(curr_b, curr_lam)
			delta = abs(oldLoss - loss)
			oldLoss = loss
			
			if i % 1000 == 0: print(curr_k1, curr_b, loss, delta)
			if not flag and delta <= 1e-20:
				print('CONVERGED')
				flag = True

		raise Exception('test')

		# parameters
		# print('k_0: ', sess.run(k_0))
		# print('k_1: ', sess.run(k_1))
		# print('theta', self.theta)

		return sess.run(k_1)

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