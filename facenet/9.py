import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import facenet

image_sizeH = 200 
image_sizeW = 200
modeldir = '/home/ubuntu/webface/FaceRecognition/code/20170512-110547/20170512-110547.pb' 
path = '../data/'
image_name1 = path+'45262.jpg' 
image_name2 = path+'1048.jpg' 


print('建立facenet embedding模型')
tf.Graph().as_default()
sess = tf.Session()

facenet.load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

print('facenet embedding模型建立完毕')
pos = 0
neg = 0
with open('../test_lst.csv') as file:
	with open('./res.txt','w') as res:
		while 1:
			line = file.readline().split(' ')
			if not line:
				break
			if len(line) < 2:
				break
			image_name1= path+ line[0]
			image_name2 = path + line[1][:-1]
			scaled_reshape = []
			image1 = scipy.misc.imread(image_name1, mode='RGB')
			image1 = cv2.resize(image1, (image_sizeH, image_sizeW), interpolation=cv2.INTER_CUBIC)
			image1 = facenet.prewhiten(image1)
			scaled_reshape.append(image1.reshape(-1,image_sizeH,image_sizeW,3))
			emb_array1 = np.zeros((1, embedding_size))
			emb_array1[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False })[0]

			image2 = scipy.misc.imread(image_name2, mode='RGB')
			image2 = cv2.resize(image2, (image_sizeH, image_sizeW), interpolation=cv2.INTER_CUBIC)
			image2 = facenet.prewhiten(image2)
			scaled_reshape.append(image2.reshape(-1,image_sizeH,image_sizeW,3))
			emb_array2 = np.zeros((1, embedding_size))
			emb_array2[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1], phase_train_placeholder: False })[0]

			dist = np.sqrt(np.sum(np.square(emb_array1[0]-emb_array2[0])))
			if(dist>1.1):
				neg +=1
				d = -1
				res.write("-1\n")
			else:
				pos +=1
				d = 1
				res.write("1\n")
			print("%s %s：%f %f: neg%d pos%d"%(line[0],line[1][:-1],dist,d,neg,pos))
	res.close()
print("neg:%d pos:%d"%(neg,pos))
