import steeldefect.training as sd_training
import steeldefect.util as sd_util
import steeldefect.losses as sd_losses
import keras
import os

def doTraining():
    labelFilePath = os.path.join(sd_util.DATA_DIR, "train.csv")
    label_df = sd_util.loadLabels(labelFilePath)
    trainFiles, validFiles = sd_training.splitTrainingDataSet(os.path.join(sd_util.PREPROCESSED_FOLDER, "AllFiles.txt"))

    # This two line for fast run training to verify code logic.
    if sd_util.FAST_VERIFICATION:
        trainFiles = trainFiles[:8]
        validFiles = validFiles[:8]

    optimizer = keras.optimizers.Adam()
    modelName = "UNET_vgg19_800_128_4_10152258"
    model = None

    for _ in range(1):
        model, modelName = sd_training.train(sd_util.MODEL_FOLDER, modelName, label_df, 
                                 trainFiles, validFiles, shrink = 2, 
                                 batch_size = 8, epoch = 2, optimizer = optimizer, lossfunc = sd_losses.diceBCELoss, 
                                 continueTraining = True, unlockResnet = "vgg19", loadedModel = model)

def doPredicting():
    pass

dataDir = "C:/Work/gitsrc/Kaggle/severstal-steel-defect-detection"
modelDir = os.path.join(dataDir, "models")
preprocessedDir = "C:/Work/gitsrc/ml/Kaggle/SteelDefects/data"
sd_util.initContext(dataDir, modelDir, preprocessedDir)
sd_util.FAST_VERIFICATION = True
doTraining()