from __future__ import print_function
import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp",one_hot=True)
X = tf.placeholder("float",[None,28*28])
Y = tf.placeholder("float",[None, 10])

w1 = tf.Variable(tf.zeros([28*28,256]))
b1 = tf.Variable(tf.zeros([256]))
l1 = tf.add(tf.matmul(X,w1),b1)

w2 = tf.Variable(tf.zeros([256,256]))
b2 = tf.Variable(tf.zeros([256]))
l2 = tf.add(tf.matmul(l1,w2),b2)

w3 = tf.Variable(tf.zeros([256,10]))
b3 = tf.Variable(tf.zeros([10]))
pred = tf.nn.softmax(tf.add(tf.matmul(l2,w3),b3))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y,logits = pred))
opt = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(pred,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
	for epoch in range(20000):
		batch = mnist.train.next_batch(128)
		sess.run(opt,feed_dict={X:batch[0],Y:batch[1]})
		if epoch % 100 ==0:
			training_accuracy = sess.run(accuracy,feed_dict={X:batch[0],Y:batch[1]})
			print("step: %d, training_accuracy : %f"%(epoch,training_accuracy))
	print(sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
