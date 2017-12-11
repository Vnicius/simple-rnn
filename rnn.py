#!/usr/bin/python3
# -*- coding : utf-8 -*-

'''
    Author: Vinícius Matheus
    Github: Vnicius
    Based in the tutorial video: https://www.youtube.com/watch?v=dFARw8Pm0Gk
    The code get the dataset mnist of TensorFlow to train a Recurrent Neural Network 
    with dimensions defined by parameters.
'''

import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# 28 x 28 images
# 28 chucks each one with 28 of size
CHUCK_SIZE = 28
NUM_CHUCK = 28
N_CLASSES_OUTPUT = 10 # 0-9 values

def load_data():
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

    return mnist

def rnn_model(x, rnn_size):
    layer = {'weigths' : tf.Variable(tf.random_normal([rnn_size, N_CLASSES_OUTPUT])),
             'biases' : tf.Variable(tf.random_normal([N_CLASSES_OUTPUT]))}

    x = tf.transpose(x, [1, 0, 2])  # transpose the matrix to another dimension
    x = tf.reshape(x, [-1, CHUCK_SIZE]) # reshape the matrix to the chuck size
    x = tf.split(x, NUM_CHUCK, 0)   # split the matrix of chucks in the number of chuncks

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # (input_data * weigths) + biases    
    output = tf.add(tf.matmul(outputs[-1], layer['weigths']), layer['biases'])

    return output

def train_nn(x, y, rnn_size, number_epochs, batch_size):
    prediction = rnn_model(x, rnn_size)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochs = number_epochs
    mnist = load_data()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                current_x, current_y = mnist.train.next_batch(batch_size)

                # reshape the data
                current_x = current_x.reshape([batch_size, NUM_CHUCK, CHUCK_SIZE])

                _, epoch_cost = sess.run([optimizer, cost],
                                         feed_dict={x : current_x, y : current_y})

                loss += epoch_cost

            print("\nEpoch: " + str(epoch + 1) + " of " + str(number_epochs) + "\nLoss: " + str(loss))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        # reshape each input image as 28 chuncks of size 28
        print("\nAccuracy: " + str(accuracy.eval({x : mnist.test.images.reshape((-1, NUM_CHUCK, CHUCK_SIZE)),
                                                  y : mnist.test.labels}) * 100) + "%")

if __name__ == '__main__':

    rnn_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    

    x = tf.placeholder('float', [None, NUM_CHUCK, CHUCK_SIZE])
    y = tf.placeholder('float', [None, N_CLASSES_OUTPUT])

    train_nn(x, y, rnn_size, num_epochs, batch_size)