import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
learning_rate = 0.01
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
#plt.plot(train_X,train_Y,'o')
#plt.show()
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(0.31,tf.float32)
b = tf.Variable(np.random.randn())

predict = tf.add(tf.multiply(w,x),b)
loss = tf.reduce_sum(tf.pow(predict- y,2))/(train_X.shape[0])
opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = opt.minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	
	sess.run(init)
	step = 1000
	for i in range(step):
		for (x_,y_) in zip(train_X,train_Y):
			sess.run(train_step,feed_dict={x:x_,y:y_})
		if i % 10 == 0:
			print("After %d iteration:"%i)
			print("w:%f"%sess.run(w))
			print("b:%f"%sess.run(b))
			c = sess.run(loss,feed_dict={x:x_,y:y_})
			print("loss:%f"%c)
			if c<0.000001 :
				break
	print("Optimization Finished!")
	plt.plot(train_X,train_Y,'o')
	plt.plot(train_X,sess.run(w)*train_X+sess.run(b))
	plt.show()