
import numpy as np
from os.path import join
image_dir=r'C:\Users\TEJA\Desktop\tensorflow'
image_paths=[join(image_dir,img) for img in ['IMG1.jpg','IMG2.jpg','IMG3.jpg','IMG4.jpg','IMG5.jpg','IMG6.jpg','IMG7.jpg','IMG8.jpg','IMG9.jpg','IMG10.jpg','IMG11.jpg','IMG12.jpg','IMG13.jpg','IMG14.jpg','IMG15.jpg','IMG16.jpg','IMG17.jpg','IMG18.jpg','IMG19.jpg','IMG20.jpg','IMG21.jpg','IMG22.jpg','IMG23.jpg','IMG24.jpg']]




from keras.preprocessing.image import load_img, img_to_array


image_size = 28

def images_to_array(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return img_array


x_train=images_to_array(image_paths)
x_train

test_image_dir=r'C:\Users\TEJA\Desktop\tensorflow'
test_image_paths=[join(image_dir,img) for img in ['TEST1.jpg','TEST2.jpg','TEST3.jpg','TEST4.jpg']]
test_image_paths


x_test=images_to_array(test_image_paths)
x_test

y=np.array([1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2])
y_test=np.array([1,1,2,2])

import pandas as pd
y_train=pd.get_dummies(y)
y

y_test=pd.get_dummies(y_test)
y_test






import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)    
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)    
    return tf.Variable(initial)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    conv=tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv+ b)
def full_layer(input, size):
    in_size = int(input.get_shape()[1])    
    W = weight_variable([in_size, size])    
    b = bias_variable([size])    
    return tf.matmul(input, W) + b 
x = tf.placeholder(tf.float32, shape=[None, 28,28,3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
x_image = tf.reshape(x, [-1, 28, 28, 3]) 
conv1 = conv_layer(x_image, shape=[5, 5, 3, 32]) 
conv1_pool = max_pool_2x2(conv1)
conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64]) 
conv2_pool = max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64]) 
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
keep_prob = tf.placeholder(tf.float32) 
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
y_conv = full_layer(full1_drop, 2)


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_)) 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

STEPS=1000
import numpy as np
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(STEPS):
        
        if i % 100 == 0:
            
            
            train_accuracy = sess.run(accuracy, feed_dict={x: x_train,y_: y_train,keep_prob: 1.0})
            print('train accuracy is ',train_accuracy)            
        sess.run(train_step, feed_dict={x: x_train, y_: y_train,keep_prob: 0.5})
    
    test_accuracy = np.mean([sess.run(accuracy,
                                      feed_dict={x:x_test, y_:y_test,keep_prob:1.0})]) 
                                      
    print(test_accuracy)
