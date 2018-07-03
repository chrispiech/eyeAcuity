import tensorflow as tf
import numpy as np
import math
import random
import scipy.stats as stats

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
		
		return min(0.9, 0.85 + (paramIndex * 0.02))
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
		return self.exponentialFit()

	def resetBeta(self):
		self.n = 0
		self.a = 0.5
		self.b = 0.5

	def lambdaFromK1(self, b_start):
		k1 = self.getCurrSize()
		return -np.log(0.2)/(k1 - b_start)

	def exponentialFit(self):
		X,y = self.reformatData()
		
		# variables
		label = tf.placeholder(tf.float32)
		x = tf.placeholder(tf.float32)

		b_start = self.getCurrSize() - 1
		lam_start = self.lambdaFromK1(b_start)


		

		k1_start = self.calcInv(b_start, lam_start)
		# print(f'True k1: {self.getCurrSize()}, k1 start: {k1_start}')
		print('---> ', b_start,  k1_start)

		
		b = tf.Variable([b_start], tf.float32)
		lam = tf.Variable([lam_start], tf.float32)

		# network
		px = 1 - tf.exp(-lam * (x - b))
		p = tf.maximum(px, FLOOR)
		p = tf.clip_by_value(px,MIN_P,MAX_P)
		networkLoss = tf.reduce_mean(-(label * tf.log(p) + (1. - label) * tf.log(1. - p)))

		# loss_summary = tf.summary.scalar("networkLoss", networkLoss)

		# optimizer
		# optimizer = tf.train.AdamOptimizer(5e-4)
		optimizer = tf.train.AdamOptimizer(5e-4)
		trainer = optimizer.minimize(networkLoss)

		# session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)

		# summary_writer = tf.summary.FileWriter(f'./logs/{self.nIterations}/train', sess.graph)

		# merge = tf.summary.merge_all()

		k1s = []
		bs = []
		losses = []

		step = 0
		LOG_EVERY = 10

		while not self.isConverged(k1s, bs, losses):
			
			output = sess.run([trainer, networkLoss], feed_dict={x:X, label:y})
			loss = output[1]

			curr_b = sess.run(b)[0]
			curr_lam = sess.run(lam)[0]
			curr_k1 = self.calcInv(curr_b, curr_lam)
			losses.append(loss)
			
			bs.append(curr_b)
			k1s.append(curr_k1)


			step += 1


		final_lam = sess.run(lam)[0]
		final_b = sess.run(b)[0]
		k1 = self.calcInv(final_b, final_lam)
		print('converged: ', k1)
		print()
		# parameters
		# print('k_0: ', sess.run(k_0))
		# print('k_1: ', sess.run(k_1))
		# print('theta', self.theta)

		return k1

	def isConverged(self, k1s, bs, losses):
		if len(k1s) > 0 and len(k1s) % 1000 == 0:
			print(bs[-1], k1s[-1], losses[-1])
			inp = input('Continue? (y): ')
			return inp != 'y'

		return False

		# return len(k1s) > 10000
		# HIST_LEN = 5
		# if len(k1s) < 2*HIST_LEN:
		# 	return False
	
		# first_avg = np.mean(k1s[-2*HIST_LEN:-HIST_LEN])
		# sec_avg = np.mean(k1s[-HIST_LEN:])
		
		# eps = 1e-5
		# delta = abs(first_avg - sec_avg)
		# # print(f'Diff: {abs(delta - eps):.2e}')
		# return delta < eps
		

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
		print(X)
		print(y)
		return X, y