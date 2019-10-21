import numpy as np
import keras
from keras.models import load_model
from keras import backend as K
import os
import sys
from os import listdir
from os.path import isfile, join
from scipy import signal
import cv2

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 1600
DATA_DIR = "/kaggle/input"
TEST_IMAGES = "/kaggle/input/severstal-steel-defect-detection/test_images"
TRAINED_MODEL_DIR = "trainedmodel2"
TRAINED_MODEL = "UNET_resnet34_1600_64_1_10051348.h5"

def loadImage(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def mask2rle(mask):
    mask= mask.T.flatten()
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
    
def saveTestResult(results, filepath):
    report = open(filepath, "w")
    report.write("ImageId_ClassId,EncodedPixels\n")
    for r in results:
        report.write("{}\n".format(r))
    report.close()

class TestingDataGenerator(keras.utils.Sequence):
    def __init__(self, path, testImageFiles, narrower = 1):
        super().__init__()
        self.path = path
        self.narrower = narrower

        self.testImageFiles = testImageFiles

    def __len__(self):
        return len(self.testImageFiles)
    
    def __getitem__(self, index):
        imageId = self.testImageFiles[index]
        imgH = int(IMAGE_HEIGHT / self.narrower)
        imgW = int(IMAGE_WIDTH)
        img = loadImage(os.path.join(self.path, imageId))
        if self.narrower != 1:
            img = cv2.resize(img, (IMAGE_WIDTH, int(IMAGE_HEIGHT / self.narrower)))
        img = img.reshape((imgH, imgW, 1))
        
        X = np.expand_dims(img, axis=0) - 88 # The average image grayscale is 88
        return X, imageId

def dummyMetrics(y_true, y_pred):
    return (K.sum(y_pred) + 1.0) / (K.sum(y_true) + 1.0)

def predict(modelPath, narrower = 1):
    results = []

    testImageFiles = [f for f in listdir(TEST_IMAGES) if isfile(join(TEST_IMAGES, f))]
    testgen = TestingDataGenerator(TEST_IMAGES, testImageFiles, narrower = narrower)
    model = load_model(modelPath, custom_objects={'dice_coef': dummyMetrics, "TverskyLoss": dummyMetrics, "DiceCoef": dummyMetrics})

    testlen = len(testgen)
    for i in range(testlen):
        print("{}/{}\r".format(i, testlen), end = "", file = sys.stderr )
        X, imageId = testgen[i]
        result = model.predict(X)
        masks = result > 0.5
        for i in range(4): # There are 4 channels
            if i == 2: # This is class 3
                # Repeat rows for narrower times
                rle = mask2rle(np.repeat(masks[0, :, :, 0], narrower, axis = 0))
                results.append("{}_{},{}".format(imageId, 3, rle))
            else:
                results.append("{}_{},".format(imageId, i + 1))

    saveTestResult(results, "submission.csv")
    print("Submitted")

    return

modelPath = os.path.join(DATA_DIR, TRAINED_MODEL_DIR, TRAINED_MODEL)
predict(modelPath, narrower = 4)