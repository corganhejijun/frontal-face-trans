# -*- coding: utf-8 -*- 
import os
from scipy import misc
import tensorflow as tf
import numpy as np
import copy

import facenet  # https://github.com/davidsandberg/facenet.git

def distance(image_folder, model_folder):
    print(image_folder)
    image_files = []
    for file in os.listdir(image_folder):
        image_files.append(os.path.join(image_folder, file))
    print("There is %d files in %s." % (len(image_files), image_folder))
    tmp_image_paths=copy.copy(image_files)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_list.append(img)
    images = np.stack(img_list)
    with tf.Graph().as_default():
        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(model_folder)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            
            nrof_images = len(image_files)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, image_files[i]))
            print('')
            
            # Print distance matrix
            print('Distance matrix')
            for i in range(nrof_images):
                print('%1d,' % i, end='')
            print('')
            average = -1
            array = np.array([])
            for i in range(nrof_images):
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    print('%1.4f,' % dist, end='')
                    if j > i:
                        array = np.append(array, dist)
                        if average < 0:
                            average = dist
                        else:
                            average = (average + dist) / 2
                print('')
            print('Average distance is %f' % average)
            print('Standard Deviation is %f' % np.std(array))
            return average, np.std(array)
    return None, None