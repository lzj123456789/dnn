import os
import tensorflow as tf

lfw_path = '/home/ubuntu/webface/data'
model_path = "./model.ckpt"

# model

# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # # Convolution Layer with 32 filters and a kernel size of 5
        # conv3 = tf.layers.conv2d(conv2, 128, 3, activation=tf.nn.relu)
        # # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 120)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        fc2 = tf.layers.dense(fc1, 84)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc2, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out
X = tf.placeholder("float",[None,28,28])
dropout = 0.8 # Dropout, probability to keep units
logits_test = conv_net(X, 11, dropout, reuse=False, is_training=False)
pred = tf.argmax(logits_test, 1)
saver = tf.train.Saver()
with open('/home/ubuntu/webface/test_lst.csv') as file:
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess,model_path)
		print("Model restored from file: %s" % model_path)
		print(tf.get_collection())
		while 1:
			line = file.readline().split(' ')
			if not line:
				break
			if len(line) <2:
				break
			for l in line:
				img = os.path.join(lfw_path,l)
				img = tf.read_file(img)
				img = tf.image.decode_jpeg(img,channels=3)
				img = tf.image.resize_images(img,[28,28])
				img = img*1.0/127.5 - 1.0
				print(sess.run(pred,feed_dict={X:img}))
		# img1 = os.path.join(lfw_path,line[0])
		# img2 = os.path.join(lfw_path,line[1])
		# #print(img1+" "+img2)
		# img1 = tf.read_file(img1)
		# img1 = tf.image.decode_jpeg(img1,channels=3)

