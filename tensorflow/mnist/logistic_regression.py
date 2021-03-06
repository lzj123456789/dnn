from __future__ import print_function

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
training_epochs = 20000
mnist = input_data.read_data_sets("/tmp",one_hot = True)

X = tf.placeholder("float",[None,28*28])
Y = tf.placeholder("float",[None,10])

W = tf.Variable(tf.zeros([28*28,10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.add(tf.matmul(X,W),b))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits = pred))
opt = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(pred,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
	for epoch in range(training_epochs):
		batch = mnist.train.next_batch(50)
		sess.run(opt,feed_dict={X:batch[0],Y:batch[1]})# also as opt.run(feed_dict={X:batch[0],Y:batch[1]})
		if epoch%100==0:
			train_accuracy = sess.run(accuracy,feed_dict={X:batch[0],Y:batch[1]})
			print("step %d ,training accuracy %g"%(epoch,train_accuracy))
	tmp_predt = tf.argmax(pred,1)
	predt = sess.run(tmp_predt,feed_dict={X:mnist.test.images})
	tmp_true = tf.argmax(Y,1)
	trueRes = sess.run(tmp_true,feed_dict={Y:mnist.test.labels})
	for (p,t) in zip(predt,trueRes):
		print("True Result is %d, Prediction: %d"%(t,p))
	acc = sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels})
	print(acc)