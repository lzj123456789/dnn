from __future__ import print_function

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp",one_hot = True)
trainX,trainY = mnist.train.next_batch(5000)
testX,testY = mnist.test.next_batch(200)

trX = tf.placeholder("float",[None,784])
teX = tf.placeholder("float",[784])

l1 = tf.reduce_sum(tf.abs( tf.add(trX,tf.negative(teX))),reduction_indices = 1)
pred = tf.argmin(l1,0)
init = tf.global_variables_initializer()
acc=0

with tf.Session() as sess:
	sess.run(init)
	for i in range(len(testX)):
		res = sess.run(pred,feed_dict = {trX:trainX,teX:testX[i,:]})
		print("Test: %d Prediction: %d True Class: %d"%(i,np.argmax(trainY[res]),np.argmax(testY[i])))
		if np.argmax(trainY[res])==np.argmax(testY[i]):
			acc +=1
print("accuracy: %f"%(1.0*acc/len(testX)))