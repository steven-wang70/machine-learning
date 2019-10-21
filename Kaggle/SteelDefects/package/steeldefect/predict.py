import numpy as np
import keras
from keras.models import load_model
from keras import backend as K
from . import util as sd_util
from . import losses as sd_losses
import os
from os import listdir
from os.path import isfile, join
from scipy import signal
import sys
from classification_models.common.blocks import Slice

class TestingDataGenerator(keras.utils.Sequence):
    def __init__(self, path, testImageFiles, shrink = 1):
        super().__init__()
        self.path = path
        self.testImageFiles = testImageFiles
        self.shrink = shrink

    def __len__(self):
        return len(self.testImageFiles)
    
    def __getitem__(self, index):
        imageId = self.testImageFiles[index]

        imageH = sd_util.IMAGE_HEIGHT // self.shrink
        imageW = sd_util.IMAGE_WIDTH // self.shrink
        img = sd_util.loadImage(os.path.join(self.path, imageId))
        if self.shrink != 1:
            img = sd_util.resizeImage(img, imageH, imageW)
        
        img -= 88 # The average image grayscale is 88

        # For each image, we will generate 12 test cases: Hflip x Vflip x  (normal, 15% darker, 20 brighter)
        # The official dimension to the model is [sample, H, W, 1]
        X = np.zeros((12, imageH, imageW))
        flipFlags = np.zeros(12, dtype = int)
        for flip in range(4):
            temp = img
            if flip & 1 != 0: # Vertical flip
                temp = np.flip(temp, axis = 0)
            if flip & 2 != 0: # Horizontal flip
                temp = np.flip(temp, axis = 1)

            for index, bright in enumerate([1, 0.85, 1.20]):
                X[flip * 3 + index, :, :] = temp * bright
                flipFlags[flip * 3 + index] = flip

        X = X.reshape((-1, imageH, imageW, 1))
        return X, flipFlags, imageId

def test_TrainingDataGenerator():
    testImageFiles = [f for f in listdir(sd_util.TEST_IMAGES) if isfile(join(sd_util.TEST_IMAGES, f))]
    datagen = TestingDataGenerator(sd_util.TEST_IMAGES, testImageFiles)
    print(len(datagen))
    X, flipFlags, imageId = datagen[11]
    print(X)
    print(imageId)

def postProcessPredict(result, flipFlags):
    for index in range(len(flipFlags)):
        temp = result[index, :, :, :]
        flip = flipFlags[index]
        if flip & 1 != 0: # Vertical flip
            temp = np.flip(temp, axis = 0)
        if flip & 2 != 0: # Horizontal flip
            temp = np.flip(temp, axis = 1)
        result[index, :, :, :] = temp

#    for i in range(sd_util.NUM_CLASS):
#        sd_util.pltContours(result[:, :, :, i])

    newResult = np.mean(result, axis = 0)
    return newResult

def test_predictImage(modelName, imageFolder, imageId):
    model = load_model(os.path.join(sd_util.MODEL_FOLDER, modelName + ".h5"), custom_objects = {"diceBCELoss": sd_losses.diceBCELoss})
    testgen = TestingDataGenerator(imageFolder, [imageId], shrink = 2)

    X, flipFlags, imageId = testgen[0]
    result = model.predict(X)
    result = postProcessPredict(result, flipFlags)

    for i in range(result.shape[2]):
        sd_util.plotContour(result[:, :, i])

def predictTests(imagePath, imageFiles, savePath, shrink = 1, loadedModel = None):
    results = []

    testgen = TestingDataGenerator(imagePath, imageFiles, shrink = shrink)
    if loadedModel is None:
        model = load_model(os.path.join(sd_util.MODEL_FOLDER, modelName + ".h5"))
    else:
        model = loadedModel

    for i in range(len(testgen)):
        print("{}/{}\r    ".format(i, len(testgen)), end = "", file = sys.stderr)
        X, flipFlags, imageId = testgen[i]
        result = model.predict(X)
        result = postProcessPredict(result, flipFlags)
        predMasks = result > 0.5
        if shrink != 1:
            predMasks = np.repeat(predMasks, shrink, axis = 0)
            predMasks = np.repeat(predMasks, shrink, axis = 1)
        for j in range(sd_util.NUM_CLASS):
            classId = j + 1
            rle = sd_util.mask2rle(predMasks[:, :, j])
            results.append("{}_{},{}".format(imageId, classId, rle))

    sd_util.saveTestResult(results, savePath)

    return

def generateSubmission():
    modelPath = os.path.join(sd_util.MODEL_FOLDER, "UNET_resnet34_1600_256_1_original.h5")
    testImageFiles = [f for f in listdir(sd_util.TEST_IMAGES) if isfile(join(sd_util.TEST_IMAGES, f))]
    predictTests(modelPath, sd_util.TEST_IMAGES, testImageFiles, os.path.join(sd_util.DATA_DIR, "submission.csv"), [3])

if __name__ == "__main__":
#    test_TrainingDataGenerator()
    modelPath = "D:\\steven\\gitsrc\\Kaggle\\severstal-steel-defect-detection\\models"
    modelName = "UNET_vgg19_800_128_4_10152258.h5"
#    result = test_predictImage(os.path.join(modelPath, modelName), sd_util.TRAIN_IMAGES, "b2148a422.jpg")
#    result = test_predictImage(os.path.join(modelPath, modelName), sd_util.TRAIN_IMAGES, "ed861754e.jpg")
#    result = test_predictImage(os.path.join(modelPath, modelName), sd_util.TRAIN_IMAGES, "5a8203514.jpg")
#    result = test_predictImage(os.path.join(modelPath, modelName), sd_util.TRAIN_IMAGES, "0696cfa05.jpg")
#    result = test_predictImage(os.path.join(modelPath, modelName), sd_util.TRAIN_IMAGES, "d571030d2.jpg")
    result = test_predictImage(os.path.join(modelPath, modelName), sd_util.TRAIN_IMAGES, "8d0ec9630.jpg")