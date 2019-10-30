import numpy as np
import keras
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from scipy import signal

DATA_H = DATA_W = 28
DATA_SIZE = DATA_H * DATA_W
EPOCHS = 20
MUTATION_RATE = 0.01
DATA_DIR = "../../Kaggle/DigitRecognizer/digit-recognizer"
VALID_PERCENT = 0.1

DIGIT_SIZE = 28
INPUT_SIZE = DIGIT_SIZE * DIGIT_SIZE
OUTPUT_SIZE = 10
BATCH_SIZE = 100

np.random.seed(12345)

def loadDatas(fileDir):
    # The first column is label. Data size is 28*28, row by row.
    trainData = pd.read_csv(os.path.join(fileDir, "train.csv")).to_numpy()

    # The test data does not contain label in the first column.
    testData = pd.read_csv(os.path.join(fileDir, "test.csv")).to_numpy()

    # from trainData, we split it to trainData and validationData, 20% is validation data.
    splitIndex = int(trainData.shape[0] * (1 - VALID_PERCENT))
#    np.random.shuffle(trainData)
    validationData = trainData[splitIndex:,:]
    trainData = trainData[:splitIndex, :]

    return trainData, validationData, testData

def prepareData(data, forConv = False):
    data_X = data[:, 1:]
    data_Y = data[:, :1]
    data_Y = data_Y.reshape((data_Y.shape[0]))

    # Normalize the data_X
    data_X = data_X / 256.0
    # Convert data_Y to one hot
    data_Y = np.eye(10)[data_Y]

    # If this is for CNN training, we will reshape the train data to * x 28 x 28 x 1
    if forConv:
        data_X = data_X.reshape((data_X.shape[0], DIGIT_SIZE, DIGIT_SIZE, 1))

    return data_X, data_Y

def createFunctionalModel(denseLayerSizes, convLayerSizes = None):
    counter = 1

    if convLayerSizes != None and len(convLayerSizes) > 0:
        input = Input(shape=(DIGIT_SIZE, DIGIT_SIZE, 1), name = "input")
        x = input
        for convSize in convLayerSizes:
            x = Conv2D(convSize, kernel_size = 3, padding='same', activation='relu', kernel_initializer='he_uniform', name = "conv_layer{0}".format(counter))(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            counter += 1
        x = Flatten(name = "layer{0}".format(counter))(x)
        counter += 1
    else:
        input = Input(shape = (INPUT_SIZE,), name = "input")
        x = input

    # Add fully connected dense layers
    for denseSize in denseLayerSizes:
        x = Dense(denseSize, activation = "relu", name = "dense_layer{0}".format(counter))(x)
        counter += 1
    prediction = Dense(OUTPUT_SIZE, activation = "softmax", name = "output_layer")(x)

    # Create model and compile it
    model = Model(inputs = input, outputs = prediction)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def plotHistory(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def showMatrix(matrix):
    fig, ax = plt.subplots(1)

    # Display the image
    ax.matshow(matrix)
    plt.show()

def trainModel(model):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
    history = model.fit(trainData_X, trainData_Y, batch_size = BATCH_SIZE, validation_data = (validData_X, validData_Y), 
            epochs= EPOCHS, verbose = 2, callbacks=[learning_rate_reduction])

    return  history

def reduceNetwork(model, reductionPercentage):
    trainable_weights = []
    trainable_biases = []
    trainable_layer_marks = [] # 1 is dense, 2 is conv
    DENSE_LAYER_FLAG = 1
    CONV_LAYER_FLAG = 2
    OTHER_LAYER_FLAG = 0
    trainable_layers_count = 0
    squared_weights_row = []
    squared_weights_sum = []
    rows_to_be_removed = []
    rowCount = 0
    output_layer_col_count = 0
    for layer in model.layers:
        if layer.trainable:
            params = layer.get_weights()
            if len(params) != 2:
                continue
            [weights, biases] = params
            trainable_weights.append(weights)
            trainable_biases.append(biases)
            rowCount += weights.shape[-1]
            output_layer_col_count = weights.shape[-1]
            if layer.name.startswith("dense_"):
                trainable_layer_marks.append(DENSE_LAYER_FLAG)
                swr = np.sum(np.square(weights), axis = 1)
                if len(trainable_layer_marks) > 1 and trainable_layer_marks[-2] == CONV_LAYER_FLAG:
                    # There must be a flatten layer between these two layers
                    # Get the conv filter count of the previous layer
                    filterCount = trainable_weights[-2].shape[-1]
                    # Group rows of sum with origination of the filter
                    swr2 = np.zeros((filterCount,))
                    for i in range(filterCount):
                        swr2[i] = np.sum(swr[i::filterCount])
                    swr = swr2
            elif layer.name.startswith("conv_"):
                trainable_layer_marks.append(CONV_LAYER_FLAG)
                swr = np.sum(np.square(weights), axis = (0, 1, 2))
            else:
                trainable_layer_marks.append(OTHER_LAYER_FLAG)
                swr = np.sum(np.square(weights), axis = 1)
            sws = np.sum(swr)
            squared_weights_row.append(swr)
            squared_weights_sum.append(sws)
            rows_to_be_removed.append([])

    # Remove the output layer column count from total row count
    rowCount -= output_layer_col_count
    trainable_layers_count = len(trainable_weights)
    rowsToRemove = int(rowCount * reductionPercentage)
    if rowsToRemove == 0:
        return None, None # Can no longer remove rows

    original_squared_weights_sum = squared_weights_sum.copy()

    for _ in range(rowsToRemove):
        hitWeightsIndex = -1
        hitRowIndex = -1
        smallestRatio = sys.float_info.max

        for weightIndex in range(1, trainable_layers_count):
            for rowIndex in range(len(squared_weights_row[weightIndex])):
                if rowIndex in rows_to_be_removed[weightIndex]:
                    continue

                thisRowEatio = squared_weights_row[weightIndex][rowIndex] / squared_weights_sum[weightIndex]
                if thisRowEatio < smallestRatio:
                    smallestRatio = thisRowEatio
                    hitWeightsIndex = weightIndex
                    hitRowIndex = rowIndex

        # We got the row that is smallest in the whole model
        rows_to_be_removed[hitWeightsIndex].append(hitRowIndex)
        squared_weights_sum[hitWeightsIndex] -= squared_weights_row[hitWeightsIndex][hitRowIndex]

    print("Info removed: {}".format(1 - np.prod(np.array(squared_weights_sum) / np.array(original_squared_weights_sum))))
    # Now we create a new model of reduced size
    denseLayerSizes = []
    convLayerSizes = []
    for layer in range(trainable_layers_count - 1):
        if trainable_layer_marks[layer] == OTHER_LAYER_FLAG:
            continue
        # Next layer's input/row is this layer's column, which is the size of this layer
        nextLayerInputCount = trainable_weights[layer].shape[-1]
        reducedSize = nextLayerInputCount - len(rows_to_be_removed[layer + 1])
        if trainable_layer_marks[layer] == DENSE_LAYER_FLAG:
            denseLayerSizes.append(reducedSize)
        else:
            convLayerSizes.append(reducedSize)

    newModel = createFunctionalModel(denseLayerSizes, convLayerSizes = convLayerSizes)
    trainableLayers = []
    for layer in newModel.layers:
        if layer.trainable and len(layer.get_weights()) != 0:
            trainableLayers.append(layer)

    for layIndex in range(trainable_layers_count):
        weights = trainable_weights[layIndex]
        biases = trainable_biases[layIndex]

        # Remove rows from the weights by rows_to_be_removed[layIndex]
        if layIndex > 0 and (trainable_layer_marks[layIndex] == DENSE_LAYER_FLAG and trainable_layer_marks[layIndex - 1] == CONV_LAYER_FLAG):
            # If the previous layer is conv and this layer is dense, we need to decode the row id
            filterCount = trainable_weights[layIndex - 1].shape[-1]
            condensedRowsToBeRemoved = rows_to_be_removed[layIndex]
            extractedRowsToBeRemoved = []
            for i in range(filterCount):
                for r in condensedRowsToBeRemoved:
                    extractedRowsToBeRemoved.append(i * filterCount + r)
            weights = np.delete(weights, extractedRowsToBeRemoved, axis = -2)
        else:
            weights = np.delete(weights, rows_to_be_removed[layIndex], axis = -2)
        # If this is not the last layer, remove columns and biases item according to next layer's rows_to_be_removed
        if layIndex + 1 != trainable_layers_count:
            weights = np.delete(weights, rows_to_be_removed[layIndex + 1], axis = -1)
            biases = np.delete(biases, rows_to_be_removed[layIndex + 1], axis = 0)

        # Set back parameters
        trainableLayers[layIndex].set_weights([weights, biases])

    newModel.compile(optimizer = Adam(lr = 0.0002), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return newModel

def testReduction(originalModelName):
    global trainData_X, validData_X
    model = keras.models.load_model(originalModelName)
    print("Load initial model")
    model.summary()
    result = model.evaluate(validData_X, validData_Y, verbose = 2)
    print("Evaluate loss {}, accuracy {}".format(result[0], result[1]))

    REDUCTION_PERCENTAGE = 0.05
    while True:
        model = reduceNetwork(model, REDUCTION_PERCENTAGE)
        if model is None:
            break

        model.summary()
        result = model.evaluate(validData_X, validData_Y, verbose = 2)
        print("Evaluate loss {}, accuracy {}".format(result[0], result[1]))
        history = trainModel(model)
        if history.history['val_accuracy'][-1] < 0.9:
            break

def testDenseOnly():
    global trainData_X, trainData_Y, validData_X, validData_Y
    trainData_X, trainData_Y = prepareData(trainData, forConv = False)
    validData_X, validData_Y = prepareData(validationData, forConv = False)
    print(validData_Y.shape)
    print(validData_Y[0])
    """
    model = createFunctionalModel((192, 128))
    model.summary()
    history = trainModel(model)
    model.save("test_r0.h5")
    """
    testReduction("test_r0.h5")

def testConv():
    global trainData_X, trainData_Y, validData_X, validData_Y
    trainData_X, trainData_Y = prepareData(trainData, forConv = True)
    validData_X, validData_Y = prepareData(validationData, forConv = True)
    print(validData_Y.shape)
    print(validData_Y[0])
    """
    model = createFunctionalModel((192, 128), convLayerSizes = (256, 128))
    model.summary()
    history = trainModel(model)
    model.save("test_cnn_r0.h5")
    """
    testReduction("test_cnn_r0.h5")
    
if __name__ == "__main__":
    trainData, validationData, testData = loadDatas(DATA_DIR)

    print(trainData.shape)
    print(validationData.shape)
    print(testData.shape)

    testConv()