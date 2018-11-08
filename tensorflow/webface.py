from __future__ import print_function

import tensorflow as tf
import os

# Dataset Parameters - CHANGE HERE
DATASET_PATH = '/home/ubuntu/webface/CASIA-WebFace-Align-96'

def read_images(dataset_path,batch_size):
	imagepaths,labels = list(),list()
	label = 0
	classes = sorted(os.walk(dataset_path).next()[1])
	print(classes)
	for c in classes:
		c_dir = os.path.join(dataset_path,c)
		walk = os.walk(c_dir).next()
		print(walk[2])
		for sample in walk[2]:
			if sample.endwith('jpg'):
				imagepaths.append(os.path.join(c_dir,sample))
				labels.append(label)
		label += 1
read_images(DATASET_PATH)
