import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from os.path import join
hot_image_dir=r'C:\Users\TEJA\Desktop\tensorflow'
hot_image_paths=[join(hot_image_dir,img) for img in ['IMG1.jpg','IMG2.jpg','IMG3.jpg','IMG4.jpg','IMG5.jpg','IMG6.jpg','IMG7.jpg','IMG8.jpg','IMG9.jpg','IMG10.jpg','IMG11.jpg','IMG12.jpg','IMG13.jpg','IMG14.jpg','IMG15.jpg','IMG16.jpg','IMG17.jpg','IMG18.jpg','IMG19.jpg','IMG20.jpg','IMG21.jpg','IMG22.jpg','IMG23.jpg','IMG24.jpg']]
hot_image_paths



from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_size = 28

def read_and_prep_images(hot_img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in hot_img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)


x_train=read_and_prep_images(hot_image_paths)
np.shape(x_train)

test_image_dir=r'C:\Users\TEJA\Desktop\tensorflow'
test_image_paths=[join(hot_image_dir,img) for img in ['TEST1.jpg','TEST2.jpg','TEST3.jpg','TEST4.jpg']]
test_image_paths


x_test=read_and_prep_images(test_image_paths)
x_test

y=np.array([1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2])
y_test=np.array([1,1,2,2])

import pandas as pd
y_train=pd.get_dummies(y)
y

y_test=pd.get_dummies(y_test)
y_test

learning_rate = 0.0001
epochs = 10
batch_size = 50

    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from mnist.train.nextbatch()

    # reshape the input data so that it is a 4D tensor.  The first value (-1) tells function to dynamically shape that
    # dimension based on the amount of data passed to it.  The two middle dimensions are set to the image size (i.e. 28
    # x 28).  The final dimension is 1 as there is only a single colour channel i.e. grayscale.  If this was RGB, this
    # dimension would be 3
x_shaped = tf.placeholder(tf.float32,shape=[None,28,28,3])
    # now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32,shape=[None, 2])


    # create some convolutional layers
layer1 = create_new_conv_layer(x_shaped, 3, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')

    # flatten the output ready for the fully connected output stage - after two layers of stride 2 pooling, we go
    # from 28 x 28, to 14 x 14 to 7 x 7 x,y co-ordinates, but with 64 output channels.  To create the fully connected,
    # "dense" layer, the new shape needs to be [-1, 7 x 7 x 64]
flattened = tf.reshape(layer2, [24, 7 * 7 * 64])

    # setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

    # another layer with softmax activations
wd2 = tf.Variable(tf.truncated_normal([1000, 2], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([2], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

    # add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialisation operator
init_op = tf.global_variables_initializer()

    # setup recording variables
    # add a summary to store the accuracy


    
with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(2):
                
                
                sess.run(optimiser, feed_dict={x_shaped: x_train, y: y_train})
                sess.run(accuracy,feed_dict={x_shaped:x_test, y:y_test})

            

        
        
        

    
def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]


    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    
    out_layer += bias

    
    out_layer = tf.nn.relu(out_layer)

    
    ksize = [1, pool_shape[0], pool_shape[1], 1]
   
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer


