from __future__ import print_function

import tensorflow as tf
import os

# Dataset Parameters - CHANGE HERE
DATASET_PATH = '/home/ubuntu/webface/CASIA-WebFace-Align-96'

def read_images(dataset_path,batch_size):
	imagepaths,labels = list(),list()
	label = 0
	classes = sorted(os.walk(dataset_path).next()[1])
	#print(classes)
	for c in classes:
		c_dir = os.path.join(dataset_path,c)
		walk = os.walk(c_dir).next()
		#print(walk[2])
		for sample in walk[2]:
			if sample.endswith('jpg'):
				imagepaths.append(os.path.join(c_dir,sample))
				labels.append(label)
		label += 1
	#for (x,y) in zip(imagepaths,labels):
	#	print(x+' '+str(y))
	imagepaths = tf.convert_to_tensor(imagepaths,dtype = tf.string)
	labels = tf.convert_to_tensor(labels,dtype = tf.int32)
	image,label = tf.train.slice_input_producer([imagepaths,labels],\
		shuffle = True)
	image = tf.read_file(image)
	image = tf.image.decode_jpeg(image,channels = 3)
	image = tf.image.resize_images(image, [112,96])
	image = image*1.0/127.5-1.0
	X,Y = tf.train.batch([image,label],batch_size = batch_size,\
		capacity = batch_size*8,\
		num_threads = 4 )
	return X,Y

X,Y = read_images(DATASET_PATH,80)
keep_prob = tf.placeholder(tf.float32)
#reshape(batch_size,h,w,channel)
xin = X
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
out = tf.add(tf.matmul(fc1,wfc2),bfc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y,logits = out))
opt = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	pred = tf.nn.softmax(out)
	correct_prediction = tf.equal(tf.cast(Y,tf.int64),tf.argmax(pred,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
	tf.train.start_queue_runners()
	for epoch in range(500):
		sess.run(opt,feed_dict={keep_prob:0.8})
		sess.run(accuracy,feed_dict={keep_prob:1.0})
		print("step:%d accuracy:%f"%(epoch,accuracy))

