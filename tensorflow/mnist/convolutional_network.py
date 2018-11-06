from __future__ import print_function
import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp",one_hot=True)
X = tf.placeholder("float",[None,28*28])
Y = tf.placeholder("float",[None,10])
keep_prob = tf.placeholder(tf.float32)
#reshape(batch_size,h,w,channel)
xin = tf.reshape(X,shape = [-1,28,28,1])
#h1 convolutional
wc1 = tf.Variable(tf.random_normal([5,5,1,32]))
bc1 = tf.Variable(tf.random_normal([32]))
hx1 = tf.nn.conv2d(xin,wc1,strides=[1,1,1,1],padding='SAME')
hx1 = tf.nn.bias_add(hx1,bc1)
conv1 = tf.nn.relu(hx1)
#pooling layer1
conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#h2 convolutional layer2
wc2 = tf.Variable(tf.random_normal([5,5,32,64]))
bc2 = tf.Variable(tf.random_normal([64]))
hx2 = tf.nn.conv2d(conv1,wc2,strides=[1,1,1,1],padding='SAME')
hx2 = tf.nn.bias_add(hx2,bc2)
conv2 = tf.nn.relu(hx2)
#pooling layer2
conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#full connected layer1
wfc1 = tf.Variable(tf.random_normal([7*7*64,1024])) 
bfc1 = tf.Variable(tf.random_normal([1024]))
fc1 = tf.reshape(conv2,[-1,7*7*64])
fc1 = tf.add(tf.matmul(fc1,wfc1),bfc1)
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1,keep_prob)
#full connected layer2
wfc2 = tf.Variable(tf.random_normal([1024,10]))
bfc2 = tf.Variable(tf.random_normal([10]))
pred = tf.add(tf.matmul(fc1,wfc2),bfc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_entropy_with_logits(labels = Y,logits = pred))
opt = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(pred,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
	for epoch in range(500):
		batch = mnist.train.next_batch(128)
		sess.run(opt,feed_dict={X:batch[0],Y:batch[1],keep_prob:0.75})
		if epoch%10==0:
			[training_accuracy,training_loss] = sess.run([accuracy,cross_entropy],feed_dict={X:batch[0],Y:batch[1],keep_prob:0.75})
			print("step: %d, training_accuracy %f, loss %f"%(epoch,training_accuracy,training_loss))
	print(sess.run(accuracy,feed_dict={\
		X:mnist.test.images,\
		Y:mnist.test.labels,\
		keep_prob:0.75}))