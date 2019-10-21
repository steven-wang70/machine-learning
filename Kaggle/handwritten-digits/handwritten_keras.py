import numpy as np
import keras
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

EPOCHS = 40
DATA_DIR = "../../../Kaggle/DigitRecognizer/digit-recognizer"
VALID_PERCENT = 0.1

DIGIT_SIZE = 28
INPUT_SIZE = DIGIT_SIZE * DIGIT_SIZE # 784
OUTPUT_SIZE = 10
BATCH_SIZE = 100
KFOLD_SIZE = 5

#np.random.seed(12345)
#tf.set_random_seed(54321)

def plotHistory(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
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

def plotContour(matrix):
    x = np.arange (matrix.shape[1])
    y = np.arange (matrix.shape[0])
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, matrix)

    plt.show()

def showDigit(arr):
    two_d = (np.reshape(arr[1:], (DIGIT_SIZE, DIGIT_SIZE))).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.show()

def loadDatas(fileDir):
    # The first column is label. Data size is 28*28, row by row.
    fullTrainData = pd.read_csv(os.path.join(fileDir, "train.csv")).to_numpy()
    # This line is for fast verifying logic only
#    fullTrainData = fullTrainData[: fullTrainData.shape[0] // 100]

    # The test data does not contain label in the first column.
    testData = pd.read_csv(os.path.join(fileDir, "test.csv")).to_numpy()

    # from trainData, we split it to trainData and validationData, 20% is validation data.
    splitIndex = int(fullTrainData.shape[0] * (1 - VALID_PERCENT))
    np.random.shuffle(fullTrainData)
    validationData = fullTrainData[splitIndex:,:]
    trainData = fullTrainData[:splitIndex, :]

    return trainData, validationData, testData

def prepareData(data, forConv = False):
    data_X = data[:, 1:]
    label_Y = data[:, :1]
    label_Y = label_Y.reshape((label_Y.shape[0]))

    # Normalize the data_X
    data_X = data_X / 255.0
    # Convert data_Y to one hot
    data_Y = np.eye(10)[label_Y]

    # If this is for CNN training, we will reshape the train data to * x 28 x 28 x 1
    if forConv:
        data_X = data_X.reshape((data_X.shape[0], DIGIT_SIZE, DIGIT_SIZE, 1))

    return data_X, data_Y, label_Y

# Test data does not contains label.
def prepareTestData(data, forConv = False):
    if forConv:
        data = data.reshape((data.shape[0], DIGIT_SIZE, DIGIT_SIZE, 1)) / 255.0

    return data

# Using config settings, create a Keras Funcational Model.
# The sample for densed layers is [(128, True, 0.2), ...] which means (size, batch normalization, dropout)
# The sample Convolutional layers config information like this: [(32, 3, False, True, 0), (48, 3, True, False, 0.2)]
# For each layer, the setting is (nun_filters, size_filter, batch normalization, max_pool, dropout)
def createFunctionalModel(denseLayers, convLayers, optimizer):
    counter = 1

    if convLayers != None and len(convLayers) > 0:
        input = Input(shape=(DIGIT_SIZE, DIGIT_SIZE, 1), name = "input")
        x = input
        for convSetting in convLayers:
            x = Conv2D(convSetting[0], kernel_size = convSetting[1], padding='same', activation='relu', kernel_initializer='he_uniform', name = "layer{0}".format(counter))(x)
            if convSetting[2]:
                x = BatchNormalization()(x)
            if convSetting[3]:
                x = MaxPooling2D(pool_size=(2, 2))(x)
            if convSetting[4] > 0:
                x = Dropout(convSetting[4])(x)
            counter += 1
        x = Flatten(name = "layer{0}".format(counter))(x)
        counter += 1
    else:
        input = Input(shape = (INPUT_SIZE,), name = "input")
        x = input

    # Add fully connected dense layers
    for denseSetting in denseLayers:
        x = Dense(denseSetting[0], activation = "relu", name = "layer{0}".format(counter))(x)
        if denseSetting[1]:
            x = BatchNormalization()(x)
        if denseSetting[2] > 0:
            x = Dropout(convSetting[4])(x)
        counter += 1
    prediction = Dense(OUTPUT_SIZE, activation = "softmax", name = "output")(x)

    # Create model and compile it
    model = Model(inputs = input, outputs = prediction)
    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def doTraining(model, epochs, trainX, trainY, cvX = None, cvY = None):
    # Training with image augmentation to improve training samples.
    if cvX is None:
        validationData = None
    else:
        validationData = (cvX, cvY)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range = 0.1)
    datagen.fit(trainX)
    history = model.fit_generator(datagen.flow(trainX, trainY, batch_size = BATCH_SIZE),
                    epochs = epochs,
                    validation_data = validationData,
                    verbose = 2,
                    steps_per_epoch = trainX.shape[0] // BATCH_SIZE, 
                    callbacks=[learning_rate_reduction])
#    plotHistory(history)

def trainModel(hiddenLayerSizes, convLayers = None, optimizer = "adam"):
    model = createFunctionalModel(hiddenLayerSizes, convLayers, optimizer)
    model.summary()

    print("Final round without K-fold")
    doTraining(model, EPOCHS, trainData_X, trainData_Y, validData_X, validData_Y)
    _, acc = model.evaluate(validData_X, validData_Y)

    return  model, acc

def saveTestResult(testLabel, config, acc):
    fileName = datetime.now().strftime("%m%d%H%M") + ".txt"
    filePath = os.path.join(DATA_DIR, fileName)
    report = open(filePath, "w")
    report.write("ACC: {}\n".format(config))
    report.write("ACC: {}\n".format(acc))
    report.write("ImageId,Label\n")
    for i in range(testLabel.shape[0]):
        report.write("{},{}\n".format(i + 1, testLabel[i]))
    report.close()

def runTest(denseConfig, cnnConfig):
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model, acc = trainModel(denseConfig, convLayers = cnnConfig, optimizer = optimizer)
    print("ACC: {}".format(acc))

    result = model.predict(validData_X)
    resultLabel = np.argmax(result, axis = 1)
    failedIndexes = np.where(resultLabel != validLabel_Y)

    testResult = model.predict(testData_X)
    testLabel = np.argmax(testResult, axis = 1)
    saveTestResult(testLabel, "Dense: {}, CNN config: {}".format(denseConfig, cnnConfig), acc)    

if __name__ == "__main__":
    trainData, validationData, testData = loadDatas(DATA_DIR)

    print("Train Data: {}".format(trainData.shape))
    print("Validation Data: {}".format(validationData.shape))
    print("Test Data: {}".format(testData.shape))

    trainData_X, trainData_Y, trainLabel_Y = prepareData(trainData, forConv = True)
    validData_X, validData_Y, validLabel_Y = prepareData(validationData, forConv = True)
    testData_X = prepareTestData(testData, forConv = True)

    denseConfigs = ([(128, True, 0.2)],) # size, batch normalization, dropout
    cnnConfigs = ([ (32, 3, True, False, 0), # num, size, batch normalization, maxpool, dropout
                    (48, 3, True, True, 0.2), 
                    (64, 3, True, False, 0), 
                    (96, 3, True, True, 0),
                    (128, 3, True, False, 0)
                 ], 
                 [  (32, 3, False, False, 0), 
                    (48, 3, True, True, 0.2), 
                    (64, 3, False, False, 0), 
                    (96, 3, True, True, 0.2), 
                    (128, 3, False, False, 0)
                 ], 
                 [  (64, 3, False, False, 0), 
                    (96, 3, True, True, 0.2), 
                    (128, 3, False, False, 0), 
                    (128, 3, True, True, 0.2)
                 ], 
                 [  (32, 3, False, False, 0), 
                    (32, 3, True, True, 0.2), 
                    (64, 3, False, False, 0), 
                    (64, 3, True, True, 0.2), 
                    (128, 3, True, True, 0.2)
                ],
                )
#    for c in cnnConfigs:
#        runTest(denseConfigs[0], c)

runTest(denseConfigs[0], cnnConfigs[0])
