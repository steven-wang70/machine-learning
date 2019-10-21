# The tensorflow version of digit recognizer.

import numpy as np
import pandas as pd
import tensorflow as tf
import os
#import matplotlib.pyplot as plt

EPOCHS = 10
DATA_DIR = "../../../Kaggle/DigitRecognizer/digit-recognizer"
VALID_PERCENT = 0.1

INPUT_SIZE = 784
OUTPUT_SIZE = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 100

def loadDatas(fileDir):
    # The first column is label. Data size is 28*28, row by row.
    trainData = pd.read_csv(os.path.join(fileDir, "train.csv")).to_numpy()

    # The test data does not contain label in the first column.
    testData = pd.read_csv(os.path.join(fileDir, "test.csv")).to_numpy()

    # from trainData, we split it to trainData and validationData, 20% is validation data.
    splitIndex = int(trainData.shape[0] * (1 - VALID_PERCENT))
    np.random.shuffle(trainData)
    validationData = trainData[splitIndex:,:]
    trainData = trainData[:splitIndex, :]

    return trainData, validationData, testData

def prepareData(data):
    data_X = data[:, 1:]
    data_Y = data[:, :1]
    data_Y = data_Y.reshape((data_Y.shape[0]))

    # Normalize the data_X
    data_X = data_X / 256.0
    # Convert data_Y to one hot
    data_Y = np.eye(10)[data_Y]

    return data_X, data_Y

def createFunctionalModel(hiddenLayerSizes, input, output):
    x = input

    # Add hidden layers
    totalLayerSizes = [INPUT_SIZE] + hiddenLayerSizes + [OUTPUT_SIZE]
    for i in range(len(totalLayerSizes) - 1):
        W = tf.Variable(tf.random_normal([totalLayerSizes[i], totalLayerSizes[i + 1]], stddev=0.03), name='W{0}'.format(i + 1))
        b = tf.Variable(tf.random_normal([totalLayerSizes[i + 1]]), name='b{0}'.format(i + 1))
        x = tf.add(tf.matmul(x, W), b)

        if i == len(totalLayerSizes) - 2:
            x = tf.nn.softmax(x)
        else:
            x = tf.nn.relu(x)

    prediction = x

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels = output))
    optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(cost)

    return cost, optimizer, prediction

def test(hiddenLayerSizes):
    global trainData_X
    global trainData_Y
    tf.reset_default_graph()
    input = tf.placeholder(tf.float32, [None, INPUT_SIZE], name = "input")
    output = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name = "output")
    cost, optimizer, prediction = createFunctionalModel(hiddenLayerSizes, input, output)
    correct_prediction = tf.equal(tf.argmax(prediction, axis = 1), tf.argmax(output, axis = 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#    model.summary()
    # initialize variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for ep in range(EPOCHS):
            # Run the mini batch
            # First shuffle data
            permutation = np.random.permutation(trainData_X.shape[0])
            trainData_X = trainData_X[permutation]
            trainData_Y = trainData_Y[permutation]
            loss = 0
            for i in range(0, trainData_X.shape[0], BATCH_SIZE):
                # Get pair of (X, y) of the current minibatch/chunk
                train_mini_X = trainData_X[i:i + BATCH_SIZE]
                train_mini_Y = trainData_Y[i:i + BATCH_SIZE]
                _, mini_loss = sess.run([optimizer, cost], feed_dict={input: train_mini_X, output: train_mini_Y})
                loss += mini_loss

            loss /= trainData_X.shape[0]
            acc = accuracy.eval({input: trainData_X, output: trainData_Y})
            val_acc = accuracy.eval({input: validData_X, output: validData_Y})
            # loss: 0.1035 - acc: 0.9717 - val_loss: 0.1169 - val_acc: 0.9671
            print("Epoch {}/{} loss: {:.4f} - acc: {:.4f} - val_acc: {:.4f}".format(ep, EPOCHS, loss, acc, val_acc))

    return 

if __name__ == "__main__":
    trainData, validationData, testData = loadDatas(DATA_DIR)

    print("Train Data: {}".format(trainData.shape))
    print("Validation Data: {}".format(validationData.shape))
    print("Test Data: {}".format(testData.shape))

    trainData_X, trainData_Y = prepareData(trainData)
    validData_X, validData_Y = prepareData(validationData)
    print("Validation Data Y: {}".format(validData_Y.shape))
    print(validData_Y[0])

    test([64, 32])