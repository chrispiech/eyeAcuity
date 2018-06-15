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
		return 100 + 2 * paramIndex

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
		X,y = self.reformatData()
		
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

		loss_summary = tf.summary.scalar("networkLoss", networkLoss)

		# optimizer
		# optimizer = tf.train.AdamOptimizer(5e-4)
		optimizer = tf.train.AdamOptimizer(5e-4)
		trainer = optimizer.minimize(networkLoss)

		# session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)

		summary_writer = tf.summary.FileWriter(f'./logs/{self.nIterations}/train', sess.graph)

		# merge = tf.summary.merge_all()

		k1s = []

		step = 0
		LOG_EVERY = 10

		while not self.isConverged(k1s):
			
			output = sess.run([trainer, networkLoss, loss_summary], feed_dict={x:X, label:y})
			loss = output[1]

			curr_b = sess.run(b)[0]
			curr_lam = sess.run(lam)[0]
			curr_k1 = self.calcInv(curr_b, curr_lam)
			k1s.append(curr_k1)



			if step % LOG_EVERY == 0:
				summary_writer.add_summary(output[2], step) 
				# print('-->', curr_k1)

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

	def isConverged(self, k1s):
		if len(k1s) > 0 and len(k1s) % 10000 == 0:
			print(k1s[-1])
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
		return X, y