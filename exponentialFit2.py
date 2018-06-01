import tensorflow as tf
import numpy as np

SEARCH_P = 0.7
INIT_X = 9
A_IN = 10

# for the exponential fit
MIN_P = 1e-10
MAX_P = 0.9999999
FLOOR = 0.25
C = 0.8
import random

class ExponentialFitPolicy:

	@staticmethod
	def getParamValue(paramIndex) :
		# param is number of iterations
		return 1000 + 2 * paramIndex

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
		return random.random() * 10

	def isDone(self):
		return self.n > self.nIterations

	def getAnswer(self):
		return self.exponentialFit()

	def exponentialFit(self):
		X,y = self.reformatData()
		
		#print(X,y)
		# variables
		label = tf.placeholder(tf.float32)
		x = tf.placeholder(tf.float32)
		k_0 = tf.Variable([4.], tf.float32)
		k_1 = tf.Variable([5.], tf.float32)

		# network
		px = 1 - tf.pow(1 - C, (x - k_0)/(k_1 - k_0))
		p = tf.maximum(px, FLOOR)
		p = tf.clip_by_value(px,MIN_P,MAX_P)
		networkLoss = tf.reduce_mean(-(label * tf.log(p) + (1. - label) * tf.log(1. - p)))

		# optimizer
		optimizer = tf.train.AdamOptimizer(1e-3)
		trainer = optimizer.minimize(networkLoss)

		# session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)

		oldLoss = sess.run(networkLoss, feed_dict={x:X, label:y})
		
		for i in range(10000):
			output = sess.run([trainer, networkLoss], feed_dict={x:X, label:y})
			loss = output[1]
			curr_k_0 = sess.run(k_0)
			curr_k_1 = sess.run(k_1)
			delta = abs(oldLoss - loss)
			oldLoss = loss
			
			if i % 100 == 0: print(curr_k_0[0], curr_k_1[0], loss, delta)

		raise Exception('test')

		# parameters
		# print('k_0: ', sess.run(k_0))
		# print('k_1: ', sess.run(k_1))
		# print('theta', self.theta)

		return sess.run(k_1)

	def reformatData(self):
		n = len(self.results)
		X = np.zeros(n)
		y = np.zeros(n)
		for i in range(n):
			# now x is in terms of "difficulty"
			X[i] = self.results[i][0]
			y[i] = self.results[i][1]
		return X, y