import numpy as np
import pandas as pd
from . import util as sd_util
from . import losses as sd_losses
from . import predict as sd_pred
import keras
from keras import backend as K
from keras import models
from keras import layers
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
from segmentation_models.utils import set_trainable
import os
import sys
from datetime import datetime

#np.random.seed(12345)
AUGMENTATION_FACTOR = 4

class TrainingDataGenerator(keras.utils.Sequence):
    def __init__(self, label_df, sampleFiles, shrink = 1, batch_size = 16, shuffle = False, augmentation = True):
        super().__init__()
        self.label_df = label_df
        self.sampleFiles = sampleFiles.to_numpy()
        self.shrink = shrink
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.augmentation = augmentation
        if self.augmentation:
            self.batch_size //= AUGMENTATION_FACTOR # We need to flip the image both vertically and horizontally, thus one image create 4 samples.

        self.on_epoch_end()

    def __len__(self):
        return self.sampleFiles.shape[0] // self.batch_size # We will discard the remaind part that could not make a batch
    
    def on_epoch_end(self):
        self.counter = 1
        if self.shuffle == True:
            np.random.shuffle(self.sampleFiles)
    
    def __getitem__(self, index):
        if sd_util.SHOW_PROGRESS:
            print("{}/{}     \r".format(self.counter, len(self)), end = "", file=sys.stderr)
        self.counter += 1

        # First select sample files
        if self.augmentation:
            tempIndex = index
            sf1 = self.sampleFiles[tempIndex * self.batch_size : (tempIndex + 1) * self.batch_size]
            tempIndex = (tempIndex + 1) % len(self)
            sf2 = self.sampleFiles[tempIndex * self.batch_size : (tempIndex + 1) * self.batch_size]
            tempIndex = (tempIndex + 1) % len(self)
            sf3 = self.sampleFiles[tempIndex * self.batch_size : (tempIndex + 1) * self.batch_size]
            tempIndex = (tempIndex + 1) % len(self)
            sf4 = self.sampleFiles[tempIndex * self.batch_size : (tempIndex + 1) * self.batch_size]
            selectedFiles = np.concatenate((sf1, sf2, sf3, sf4), axis = 0)
        else:
            selectedFiles = self.sampleFiles[index * self.batch_size : (index + 1) * self.batch_size]

        imageH = sd_util.IMAGE_HEIGHT // self.shrink
        imageW = sd_util.IMAGE_WIDTH // self.shrink
        X = np.empty((selectedFiles.shape[0], imageH, imageW), dtype=np.float32)
        Y = np.empty((selectedFiles.shape[0], imageH, imageW, sd_util.NUM_CLASS), dtype=np.int8)
        # Each index has 4 lines in the labels data frame
        for i in range(selectedFiles.shape[0]):
            fileName = selectedFiles[i, 0]
            selectedRows = self.label_df[self.label_df.ImageId == fileName]
            img = sd_util.loadImage(os.path.join(sd_util.TRAIN_IMAGES, fileName))
            if self.shrink != 1:
                img = sd_util.resizeImage(img, imageH, imageW)
            X[i, :, :] = img

            for classOffset in range(sd_util.NUM_CLASS):
                if selectedRows.iloc[classOffset].hasRle:
                    rle = selectedRows.iloc[classOffset].EncodedPixels
                    mask = sd_util.rle2mask(rle)
                    if self.shrink != 1:
                        mask = mask[::self.shrink, ::self.shrink]
                    Y[i, :, :, classOffset] = mask

        if self.augmentation:
            # Shift masks up/down and left/right randomly
            shiftScales = np.random.rand(X.shape[0] * 2)
            # Find margins of masks
            for i in range(X.shape[0]):
                Ymasked = np.sum(Y[i, :, :, :], axis = 2)
                zeroRows = np.where(Ymasked.any(axis = 1))[0]
                if zeroRows.shape[0] != 0:
                    topMargin = np.min(zeroRows)
                    bottonMargin = np.max(zeroRows)
                    vdisp = int((topMargin + imageH - bottonMargin) * shiftScales[ 2 * i])
                    if vdisp > topMargin: # We are shifting down
                         X[i, :, :] = np.roll(X[i, :, :], vdisp - topMargin, axis = 0)
                         X[i, :vdisp - topMargin, :] = 0
                         Y[i, :, :, :] = np.roll(Y[i, :, :, :], vdisp - topMargin, axis = 0)
                    elif vdisp < topMargin: # We are shifting up
                         X[i, :, :] = np.roll(X[i, :, :], imageH + vdisp - topMargin, axis = 0)
                         X[i, imageH + vdisp - topMargin:, :] = 0
                         Y[i, :, :, :] = np.roll(Y[i, :, :, :], imageH + vdisp - topMargin, axis = 0)
                    else: # Do not shift
                        pass
                        
                zeroColumns = np.where(Ymasked.any(axis = 0))[0]
                if zeroColumns.shape[0] != 0:
                    leftMargin = np.min(zeroColumns)
                    rightMargin = np.max(zeroColumns)
                    hdisp = int((leftMargin + imageW - rightMargin) * shiftScales[ 2 * i + 1])
                    if hdisp > leftMargin: # We are shifting down
                         X[i, :, :] = np.roll(X[i, :, :], hdisp - leftMargin, axis = 1)
                         X[i, :, :hdisp - leftMargin] = 0
                         Y[i, :, :, :] = np.roll(Y[i, :, :, :], hdisp - leftMargin, axis = 1)
                    elif hdisp < leftMargin: # We are shifting up
                         X[i, :, :] = np.roll(X[i, :, :], imageW + hdisp - leftMargin, axis = 1)
                         X[i, :, imageW + hdisp - leftMargin:] = 0
                         Y[i, :, :, :] = np.roll(Y[i, :, :, :], imageW + hdisp - leftMargin, axis = 1)
                    else: # Do not shift
                        pass

            # Determine how to flip
            flipFlag = index % 4
            if flipFlag & 1 != 0: # Vertical flip
                X = np.flip(X, axis = 1)
                Y = np.flip(Y, axis = 1)
            if flipFlag & 2 != 0: # Horizontal flip
                X = np.flip(X, axis = 2)
                Y = np.flip(Y, axis = 2)

            X -= 88 # The average grayscale of images is 88
            # Randomly change scale between 80% to 130%
            scales = np.random.rand(X.shape[0]) * 0.5 + 0.8
            for i in range(X.shape[0]):
                X[i, :, :] *= scales[i]

        X = X.reshape((X.shape[0], imageH, imageW, 1))
        return X, Y

def test_TrainingDataGenerator():
    trainFiles, validFiles = splitTrainingDataSet(os.path.join(sd_util.PREPROCESSED_FOLDER, "AllFiles.txt"))
    print(trainFiles.shape)
    datagen = TrainingDataGenerator(label_df, trainFiles, shrink = 2, augmentation = True)
    batches = len(datagen)
    for i in range(batches):
        X, Y = datagen[i]
        for j in range(X.shape[0]):
            Xtemp = X[j, :, :, 0] + 88.0
            Yc3 = Y[j, :, :, 2] * 60
            Yc4 = Y[j, :, :, 3] * 60
            Xtemp -= Yc3 + Yc4
            sd_util.showMatrix(Xtemp)

# Read file names from a file and split it.
def splitTrainingDataSet(files):
    files_df = pd.read_csv(files)
    len = files_df.shape[0]
    validCount = int(len * sd_util.VALID_SET_RATIO)
    validFiles = files_df[:validCount]
    trainFiles = files_df[validCount:]

    return trainFiles, validFiles

def printModelStructure(model, indent = 0):
    if hasattr(model, "layers"):
        for l in model.layers:
            if hasattr(l, "trainable"):
                print("{}{} {} {}".format("\t" * indent, type(l), l.name, l.trainable))
            printModelStructure(l, indent + 1)

def unfreezeClassifierNet(model, backbone):
    # The resnet34 model is a embeded layer in the current model with name "u-resnet34"
    resnet34 = model.get_layer("u-{}".format(backbone))
    for layer in resnet34.layers:
        layer.trainable = True

# We always verify model on the whole validataion set of all files
# In the verification, we will print out for each classId according to whole area:
# 1. True mask ratio
# 2. Pred mask ratio
# 3. There intersection
def compareResultAgainst(resultFile, label_df):
    result_df = sd_util.loadLabels(resultFile)

    # The data structure to store result is: [[true, pred, intersect]...]
    veriResult = np.zeros((sd_util.NUM_CLASS, 3), dtype = int)

    for _, row in result_df.iterrows():
        trueRle = None
        predRle = None
        classIndex = row["ClassId"] - 1
        if row["hasRle"] == True:
            predRle = row["EncodedPixels"]
            veriResult[classIndex, 1] += sd_util.getRleArea(predRle)

        
        trueResult = label_df[label_df.ImageId_ClassId == row["ImageId_ClassId"]]
        if trueResult.iloc[0].hasRle:
            trueRle = trueResult.iloc[0].EncodedPixels
            veriResult[classIndex, 0] += sd_util.getRleArea(trueRle)

        if trueRle is not None and predRle is not None:
            trueMask = sd_util.rle2mask(trueRle)
            predMask = sd_util.rle2mask(predRle)
            veriResult[classIndex, 2] += np.sum(trueMask * predMask)

    totalPoints = sd_util.IMAGE_HEIGHT * sd_util.IMAGE_WIDTH * result_df.shape[0] * 1.0 / sd_util.NUM_CLASS
    for i in range(sd_util.NUM_CLASS):
        score = 1.0 * (veriResult[i, 0] + veriResult[i, 1] - 2 * veriResult[i, 2] + 1e-6) / (veriResult[i, 0] + 1e-6)
        print("ClassID= {}, True= {}, Pred= {}, Intersection= {}, Score= {}".
            format(i + 1, veriResult[i, 0] / totalPoints, veriResult[i, 1] / totalPoints, veriResult[i, 2] / totalPoints, score))

def verifyModel(model, label_df, shrink = 1):
    _, validFiles = splitTrainingDataSet(os.path.join(sd_util.PREPROCESSED_FOLDER, "AllFiles.txt"))
    if sd_util.FAST_VERIFICATION:
        validFiles = validFiles[:10]
    validFiles = validFiles.to_numpy().reshape((validFiles.shape[0],))
    if sd_util.FAST_VERIFICATION:
        validFiles = validFiles[:30]

    sd_pred.predictTests(sd_util.TRAIN_IMAGES, validFiles, "testResult.csv", shrink = shrink, loadedModel = model)
    compareResultAgainst("testResult.csv", label_df)

def train(modelName, label_df, trainFiles, validFiles, batch_size = 16, shrink = 1, epoch = 10, optimizer = None,
            lossfunc = "binary_crossentropy", continueTraining = False, unlockResnet = None, loadedModel = None):
    traingen = TrainingDataGenerator(label_df, trainFiles, batch_size = batch_size, shrink = shrink, shuffle = True, augmentation = sd_util.AUGMENTATION)
    validgen = TrainingDataGenerator(label_df, validFiles, batch_size = batch_size, shrink = shrink, augmentation = sd_util.AUGMENTATION)
    print("")
    print("Train start at: {}".format(datetime.now()))

    filepathname = os.path.join(sd_util.MODEL_FOLDER, modelName + ".h5")

    if loadedModel is None:
        model = load_model(filepathname, custom_objects = {lossfunc.__name__: lossfunc})
    else:
        model = loadedModel

    if continueTraining:
        if unlockResnet is not None:
            unfreezeClassifierNet(model, unlockResnet)
            model.compile(optimizer = optimizer, loss = lossfunc, metrics = ["accuracy"])
    else:
        model.compile(optimizer = optimizer, loss = lossfunc, metrics = ["accuracy"])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

    print("From Model: {}, ContinueTraining: {}, UnlockResnet: {}".format(modelName, continueTraining, unlockResnet))
    # Training model
    _ = model.fit_generator(traingen, validation_data = validgen, epochs = epoch, verbose=2, callbacks=[learning_rate_reduction]) # Return history
    # Save model
    timString = datetime.now().strftime("%m%d%H%M")
    newModelName = modelName[0:modelName.rfind("_") + 1] + timString
    filename = "{}.h5".format(newModelName)
    print("Save H5 file to: {}".format(filename))
    model.save(os.path.join(sd_util.MODEL_FOLDER, filename))
    print("Model Saved.")
    # Verify model
    verifyModel(model, label_df, shrink = shrink)

    return model, newModelName

def trainWithLossFunction():
    trainFiles, validFiles = splitTrainingDataSet(os.path.join(sd_util.PREPROCESSED_FOLDER, "AllFiles.txt"))

    # This two line for fast run training to verify code logic.
    if sd_util.FAST_VERIFICATION:
        trainFiles = trainFiles[:30]
        validFiles = validFiles[:30]

    optimizer = keras.optimizers.Adam()
    modelName = "UNET_vgg19_800_128_4_10170225"
    model = None

    for _ in range(10):
        model, modelName = train(sd_util.MODEL_FOLDER, modelName, label_df, 
                                 trainFiles, validFiles, shrink = 2, 
                                 batch_size = 8, epoch = 3, optimizer = optimizer, lossfunc = sd_losses.diceBCELoss, 
                                 continueTraining = True, unlockResnet = "vgg19", loadedModel = model)

if __name__ == "__main__":
    label_df = sd_util.loadLabels()
#    test_TrainingDataGenerator()
    trainWithLossFunction()
