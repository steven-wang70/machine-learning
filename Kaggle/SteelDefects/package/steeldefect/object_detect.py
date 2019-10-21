import numpy as np
import pandas as pd
import util as sd_util
import losses as sd_losses
import predict as sd_pred
import keras
from keras import backend as K
from keras import models
from keras import layers
from keras.models import load_model
from keras.optimizers import adam
from keras.callbacks import ReduceLROnPlateau
from segmentation_models.utils import set_trainable
import os
import sys
from datetime import datetime
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

np.random.seed(12345)
AUGMENTATION_FACTOR = 4
#tf.compat.v1.enable_eager_execution()

class ODTrainingDataGenerator(keras.utils.Sequence):
    def __init__(self, boxlabel_df, sampleFiles, shrink = 1, batch_size = 16, shuffle = False, augmentation = True):
        super().__init__()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.augmentation = augmentation
        if self.augmentation:
            self.batch_size //= AUGMENTATION_FACTOR # We need to flip the image both vertically and horizontally, thus one image create 4 samples.

        # We are using 25 x 4 grid
        self.IMAGE_H = sd_util.IMAGE_HEIGHT // shrink
        self.IMAGE_W = sd_util.IMAGE_WIDTH // shrink
        self.GRID_COLUMN = 25
        self.GRID_ROW = 4
        self.GRID_COUNT = self.GRID_COLUMN * self.GRID_ROW
        self.GRID_WIDTH = sd_util.IMAGE_WIDTH // self.GRID_COLUMN
        self.GRID_HEIGHT = sd_util.IMAGE_HEIGHT // self.GRID_ROW
        self.CONF_OFFSET = 0
        self.CLASS_OFFSET = 1
        self.COORD_OFFSET = 5
        self.SIZE_OFFSET = 7
        self.CELL_INFO_SIZE = 9

        self.preprocess(boxlabel_df, sampleFiles, shrink)
        self.imageIndexes = np.arange(len(sampleFiles))

        self.on_epoch_end()

    def __len__(self):
        return self.imageIndexes.shape[0] // self.batch_size # We will discard the remaind part that could not make a batch
    
    # Preprocess including:
    # 1. Load image and resize
    # 2. Load rle and convert to rect and resize
    # 3. Organize it to a structure for easier access
    # We will store it like:
    # [(image in np, [rects of class1,...], [margins of left, top, right, bottom]), ...]
    def preprocess(self, boxlabel_df, sampleFiles, shrink):
        # First create a list with required length
        self.dataStore = [None] * len(sampleFiles)
        sampleCount = len(sampleFiles)
        for fileIndex in range(sampleCount):
            print("Preprocess: {}/{}    \r".format(fileIndex, sampleCount), end = "", file = sys.stderr)
            fileName = sampleFiles.iloc[fileIndex].TrainImageId
            # Load and resize image
            img = sd_util.loadImage(os.path.join(sd_util.TRAIN_IMAGES, fileName))
            if shrink != 1:
                img = sd_util.resizeImage(img, self.IMAGE_H, self.IMAGE_W)
            img = img.reshape((self.IMAGE_H, self.IMAGE_W, 1))
            img -= 88 # The average grayscale of images is 88

            selectedRows = boxlabel_df[boxlabel_df.ImageId == fileName]
            defectClasses = [None] * sd_util.NUM_CLASS
            margins = [self.IMAGE_W, 0, self.IMAGE_H, 0]
            for classIndex in range(sd_util.NUM_CLASS):
                if selectedRows.iloc[classIndex].hasBox:
                    encodedBoxes = selectedRows.iloc[classIndex].EncodedBoxes
                    rects = sd_util.encoded2rects(encodedBoxes)
                    temp = np.zeros((len(rects), 4)) # Convert it from tuple to np matrix
                    temp[:, :] = rects
                    rects = temp
                    if shrink != 1:
                        rects /= shrink

                    defectClasses[classIndex] = rects
                    if np.min(rects[:, 0]) < margins[0]:
                        margins[0] = np.min(rects[:, 0])
                    if np.min(rects[:, 1]) < margins[1]:
                        margins[1] = np.min(rects[:, 1])
                    if np.max(rects[:, 2]) > margins[2]:
                        margins[2] = np.max(rects[:, 2])
                    if np.max(rects[:, 3]) > margins[3]:
                        margins[3] = np.max(rects[:, 3])

            self.dataStore[fileIndex] = (img, defectClasses, margins)

    def on_epoch_end(self):
        self.counter = 1
        if self.shuffle == True:
            np.random.shuffle(self.imageIndexes)
    
    def __getitem__(self, index):
        print("Batch index: {}/{}     \r".format(self.counter, len(self)), end = "", file=sys.stderr)
        self.counter += 1

        # First select sample files
        if self.augmentation:
            selectedImages = np.empty((0,), dtype = int)
            for i in range(AUGMENTATION_FACTOR):
                absIndex = (index + i) %  len(self)
                sf = self.imageIndexes[absIndex * self.batch_size : (absIndex + 1) * self.batch_size]
                selectedImages = np.concatenate((selectedImages, sf), axis = 0)
        else:
            selectedImages = self.imageIndexes[index * self.batch_size : (index + 1) * self.batch_size]

        X = np.zeros((selectedImages.shape[0], self.IMAGE_H, self.IMAGE_W, 1), dtype=np.float32)
        # We are using 25 x 4 grid, each grid has a box of 4 classes.
        # Each box has a confidence, central point, height and width.
        # So in this case, this layers has 900 nodes.
        # The structure is:
        # [100 confidence, 100 classe of 4, 100 [x, y], 100 [w, h]]
        # [x, y, w, h] are normalized to the cell, the center of cell is [0, 0]
        # But for convenience, we will first create a matrix of [25, 4, depth].
        # The depth is the range as [conf:1, class:4, x:1, y:1, w:1, h:1]
        Y = np.zeros((selectedImages.shape[0], self.GRID_ROW, self.GRID_COLUMN, self.CELL_INFO_SIZE), dtype=np.float32)
        # Each index has 4 lines in the labels data frame
        for i in range(selectedImages.shape[0]):
            AreaMapping = np.zeros((self.GRID_ROW, self.GRID_COLUMN))
            X[i, :, :, :] = self.dataStore[selectedImages[i]][0]

            defectClasses = self.dataStore[selectedImages[i]][1]
            for j in range(sd_util.NUM_CLASS):
                if defectClasses[j] is None:
                    continue
                rects = defectClasses[j]
                for iRect in range(len(rects)):
                    centralX = (rects[iRect][0] + rects[iRect][2]) // 2
                    centralY = (rects[iRect][1] + rects[iRect][3]) // 2
                    gridIndexRow = int(centralY // self.GRID_HEIGHT)
                    gridIndexCol = int(centralX // self.GRID_WIDTH)
                    normWidth = (rects[iRect][2] - rects[iRect][0]) / self.GRID_WIDTH
                    normHeight = (rects[iRect][3] - rects[iRect][1]) / self.GRID_HEIGHT
                    rectArea = normWidth * normHeight
                    if rectArea > AreaMapping[gridIndexRow, gridIndexCol]:
                        AreaMapping[gridIndexRow, gridIndexCol] = rectArea
                        # Fill in the class flag, confidence = 1, and coordinate
                        # [1, C1, C2, C3, C4, CX, CY, W, H]
                        Y[i, gridIndexRow, gridIndexCol, 0] = 1 # confidence
                        Y[i, gridIndexRow, gridIndexCol, self.CLASS_OFFSET : self.COORD_OFFSET] = 0 # Reset the class IDs
                        Y[i, gridIndexRow, gridIndexCol, self.CLASS_OFFSET + j] = 1 # One hot of class id

                        normCentralX = (centralX % self.GRID_WIDTH) / self.GRID_WIDTH - 0.5
                        normCentralY = (centralY % self.GRID_HEIGHT) / self.GRID_HEIGHT - 0.5
                        Y[i, gridIndexRow, gridIndexCol, self.COORD_OFFSET] = normCentralX
                        Y[i, gridIndexRow, gridIndexCol, self.COORD_OFFSET + 1] = normCentralY

                        Y[i, gridIndexRow, gridIndexCol, self.SIZE_OFFSET] = normWidth
                        Y[i, gridIndexRow, gridIndexCol, self.SIZE_OFFSET + 1] = normHeight

        if self.augmentation:
            # At last, determine how to flip
            flipFlag = index % 4
            if flipFlag & 1 != 0: # Vertical flip
                X = np.flip(X, axis = 1)
                Y = np.flip(Y, axis = 1)
                Y[:, :, :, self.SIZE_OFFSET + 1] = -Y[:, :, :, self.SIZE_OFFSET + 1]
            if flipFlag & 2 != 0: # Horizontal flip
                X = np.flip(X, axis = 2)
                Y = np.flip(Y, axis = 2)
                Y[:, :, :, self.SIZE_OFFSET] = -Y[:, :, :, self.SIZE_OFFSET]

            # We could also shift the image left or right            

        Y_confid = Y[:, :, :, 0].reshape((-1, 1 * self.GRID_COUNT))
        Y_class = Y[:, :, :, self.CLASS_OFFSET : self.COORD_OFFSET].reshape((-1, 4 * self.GRID_COUNT))
        Y_coord = Y[:, :, :, self.COORD_OFFSET : self.SIZE_OFFSET].reshape((-1, 2 * self.GRID_COUNT))
        Y_size = Y[:, :, :, self.SIZE_OFFSET:].reshape((-1, 2 * self.GRID_COUNT))
        Y = np.concatenate((Y_confid, Y_class, Y_coord, Y_size), axis = 1)
        return X, Y

def test_ODTrainingDataGenerator():
    trainFiles, validFiles = splitTrainingDataSet(os.path.join(sd_util.PREPROCESSED_FOLDER, "AllFiles.txt"))
    trainFiles = trainFiles[:20]
    print(trainFiles.shape)
    datagen = ODTrainingDataGenerator(label_df, trainFiles, shrink = 2)
    print(len(datagen))
    X, Y = datagen[6]
    print(X)
    np.set_printoptions(threshold=sys.maxsize)
    print(Y)
    for data in datagen.dataStore:
        for defect in data[1]:
            if defect is None:
                continue
            sd_util.showMatrixWithRects(data[0][:, :, 0] + 88, defect[0])

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

def unfreezeResnet(model):
    # The resnet34 model is a embeded layer in the current model with name "u-resnet34"
    for layer in model.layers:
        layer.trainable = True

GRID_COL = 25
GRID_ROW = 4
GRID_COUNT = GRID_COL * GRID_ROW
NUM_CLASS = 4
BOX_PER_CELL = 1
BOX_COORD = 2
BOX_SIZE = 2

CLASS_OFFSET = GRID_COUNT * BOX_PER_CELL
COORD_OFFEST = GRID_COUNT * (BOX_PER_CELL + NUM_CLASS)
SIZE_OFFSET = GRID_COUNT * (BOX_PER_CELL + NUM_CLASS + BOX_COORD)

# The formula comes from https://github.com/makatx/YOLO_ResNet
def loss_loop_body(t_true, t_pred, i, ta):
    '''
    This funtion is the main body of the custom_loss() definition, called from within the tf.while_loop()
    The loss funtion implemented here is as decsribed in the original YOLO paper: https://arxiv.org/abs/1506.02640
    # Arguments
    t_true: the ground truth tensor; shape: (batch_size, 1573)
    t_pred: the predicted tensor; shape: (batch_size, 1573)
    i: iteration cound of the while_loop
    ta: TensorArray that stores loss
    '''

    lambda_obj = 1.0
    lambda_noobj = 0.012
    lambda_class = 1.0
    lambda_coord = 4.5
    total_loss = 0

    ### Get the current iteration's tru and predicted tensor
    c_true = t_true[i]
    c_pred = t_pred[i]

    # Confidence loss when there is objects
    conf_true = c_true[ : CLASS_OFFSET]
    conf_pred = c_pred[ : CLASS_OFFSET]
    conf_loss_obj = K.sum(conf_true * K.square(1 - conf_pred))
    conf_loss_noobj = K.sum((1 - conf_true) * K.square(conf_pred))

    # Class prediction loss
    class_true = c_true[CLASS_OFFSET : COORD_OFFEST]
    class_true = K.reshape(class_true, (GRID_COUNT, NUM_CLASS))
    class_pred = c_pred[CLASS_OFFSET : COORD_OFFEST]
    class_pred = K.reshape(class_pred, (GRID_COUNT, NUM_CLASS))
    class_loss =  K.sum(conf_true * K.sum(K.square(class_true - class_pred), axis = 1))

    # Distance loss
    x_true = c_true[COORD_OFFEST : SIZE_OFFSET : BOX_COORD]
    x_pred = c_pred[COORD_OFFEST : SIZE_OFFSET : BOX_COORD]
    y_true = c_true[COORD_OFFEST + 1 : SIZE_OFFSET : BOX_COORD]
    y_pred = c_pred[COORD_OFFEST + 1 : SIZE_OFFSET : BOX_COORD]
    distance_loss = K.sum(conf_true * (K.square(x_true - x_pred) + K.square(y_true - y_pred)))

    # Box size loss
    w_true = c_true[SIZE_OFFSET : : BOX_SIZE]
    w_pred = c_pred[SIZE_OFFSET : : BOX_SIZE]
    h_true = c_true[SIZE_OFFSET + 1 : : BOX_SIZE]
    h_pred = c_pred[SIZE_OFFSET + 1 : : BOX_SIZE]
    box_loss = K.sum(conf_true * (K.square(K.sqrt(w_true) - K.sqrt(w_pred)) + K.square(K.sqrt(h_true) - K.sqrt(h_pred))))

    total_loss += lambda_obj * conf_loss_obj + lambda_noobj * conf_loss_noobj + lambda_class * class_loss + lambda_coord * (distance_loss + box_loss)

    ta = ta.write(i, total_loss)
    return t_true, t_pred, i + 1, ta

def yolo_loss(y_true, y_pred):
    c = lambda t, p, i, ta : K.less(i, K.shape(t)[0])
    ta = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    #ta_debug = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

    ### tf.while_loop creates a Tensorflow map with our loss function calculation (in loop_body())
    t, p, i, ta = tf.while_loop(c, loss_loop_body, [y_true, y_pred, 0, ta])

    ### convert TensorArray into a tensor and calculate mean loss
    loss_tensor = ta.stack()
    loss_mean = K.mean(loss_tensor)

    return loss_mean


def confidence_metric(c_true, c_pred):
    threshold = 0.5

    # Confidence metrics
    conf_true = c_true[ : CLASS_OFFSET]
#    conf_true = tf.Print(conf_true, [conf_true], message="conf_true=", summarize = -1)
    conf_pred = c_pred[ : CLASS_OFFSET]
    conf_pred = K.cast(K.greater(conf_pred, threshold), 'float32')
    intersect_mask = conf_true * conf_pred
    conf_true, conf_pred = K.sum(conf_true), K.sum(conf_pred)

    return conf_true, conf_pred, intersect_mask

def metrics_confid_only_loop(t_true, t_pred, i, ta_true, ta_pred, ta_intersect):
    ### Get the current iteration's tru and predicted tensor
    c_true = t_true[i]
    c_pred = t_pred[i]

    conf_true, conf_pred, intersect_mask = confidence_metric(c_true, c_pred)
    intersect_mask = K.sum(intersect_mask)

    ta_true = ta_true.write(i, conf_true)
    ta_pred = ta_pred.write(i, conf_pred)
    ta_intersect = ta_intersect.write(i, intersect_mask)
    return t_true, t_pred, i + 1, ta_true, ta_pred, ta_intersect

def yolo_metric_confid(y_true, y_pred):
    c = lambda t, p, i, ta_true, ta_pred, ta_intersect : K.less(i, K.shape(t)[0])
    ta_true = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    ta_pred = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    ta_intersect = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

    # tf.while_loop creates a Tensorflow map with our loss function calculation (in loop_body())
    t, p, i, ta_true, ta_pred, ta_intersect = tf.while_loop(c, metrics_confid_only_loop, [y_true, y_pred, 0, ta_true, ta_pred, ta_intersect])

    ### convert TensorArray into a tensor and calculate mean metrics
    ta_true_tensor = ta_true.stack()
    ta_pred_tensor = ta_pred.stack()
    ta_intersect_tensor = ta_intersect.stack()
    confid_tensor = 2 * ta_intersect_tensor / (ta_true_tensor + ta_pred_tensor + 1e-4)
    confid = K.mean(confid_tensor)

    return confid

def class_metric(c_true, c_pred):
    # Class prediction metrics
    class_true = c_true[CLASS_OFFSET : COORD_OFFEST]
    class_true = K.reshape(class_true, (GRID_COUNT, NUM_CLASS))
    class_pred = c_pred[CLASS_OFFSET : COORD_OFFEST]
    class_pred = K.reshape(class_pred, (GRID_COUNT, NUM_CLASS))
    class_pred = K.one_hot(K.argmax(class_pred, axis = 1), num_classes = NUM_CLASS)
    intersect_mask = K.sum(class_true * class_pred, axis = 1)

    return intersect_mask

def metrics_class_only_loop(t_true, t_pred, i, ta_intersect):
    ### Get the current iteration's tru and predicted tensor
    c_true = t_true[i]
    c_pred = t_pred[i]

    intersect_mask = class_metric(c_true, c_pred)
    intersect_mask = K.sum(intersect_mask)

    ta_intersect = ta_intersect.write(i, intersect_mask)
    return t_true, t_pred, i + 1, ta_intersect

def yolo_metric_class(y_true, y_pred):
    c = lambda t, p, i, ta_intersect : K.less(i, K.shape(t)[0])
    ta_intersect = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

    # tf.while_loop creates a Tensorflow map with our loss function calculation (in loop_body())
    t, p, i, ta_intersect = tf.while_loop(c, metrics_class_only_loop, [y_true, y_pred, 0, ta_intersect])

    ### convert TensorArray into a tensor and calculate mean metrics
    ta_intersect_tensor = ta_intersect.stack()
    class_mean = K.mean(ta_intersect_tensor)

    return class_mean

def coord_metric(c_true, c_pred):
    # IoU
    x_true = c_true[COORD_OFFEST : SIZE_OFFSET : BOX_COORD]
    x_pred = c_pred[COORD_OFFEST : SIZE_OFFSET : BOX_COORD]
    y_true = c_true[COORD_OFFEST + 1 : SIZE_OFFSET : BOX_COORD]
    y_pred = c_pred[COORD_OFFEST + 1 : SIZE_OFFSET : BOX_COORD]
    w_true = c_true[SIZE_OFFSET : : BOX_SIZE]
    w_pred = c_pred[SIZE_OFFSET : : BOX_SIZE]
    h_true = c_true[SIZE_OFFSET + 1 : : BOX_SIZE]
    h_pred = c_pred[SIZE_OFFSET + 1 : : BOX_SIZE]

    x_dist = K.abs(x_true - x_pred)
    y_dist = K.abs(y_true - y_pred)

    ### (w1/2 +w2/2 -d) > 0 => intersection, else no intersection
    ### (h1/2 +h2/2 -d) > 0 => intersection, else no intersection
    wwd = K.relu(w_true/2 + w_pred/2 - x_dist)
    hhd = K.relu(h_true/2 + h_pred/2 - y_dist)

    area_true = K.sum(w_true * h_true)
    area_pred = K.sum(w_pred * h_pred)
    area_intersect = wwd * hhd

    return area_true, area_pred, area_intersect

def metrics_coord_only_loop(t_true, t_pred, i, ta_true, ta_pred, ta_intersect):
    ### Get the current iteration's tru and predicted tensor
    c_true = t_true[i]
    c_pred = t_pred[i]

    area_true, area_pred, area_intersect = coord_metric(c_true, c_pred)
    area_intersect = K.sum(area_intersect)

    ta_true = ta_true.write(i, area_true)
    ta_pred = ta_pred.write(i, area_pred)
    ta_intersect = ta_intersect.write(i, area_intersect)
    return t_true, t_pred, i + 1, ta_true, ta_pred, ta_intersect

def yolo_metric_coord(y_true, y_pred):
    c = lambda t, p, i, ta_true, ta_pred, ta_intersect : K.less(i, K.shape(t)[0])
    ta_true = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    ta_pred = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    ta_intersect = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

    # tf.while_loop creates a Tensorflow map with our loss function calculation (in loop_body())
    t, p, i, ta_true, ta_pred, ta_intersect = tf.while_loop(c, metrics_coord_only_loop, [y_true, y_pred, 0, ta_true, ta_pred, ta_intersect])

    ### convert TensorArray into a tensor and calculate mean metrics
    ta_true_tensor = ta_true.stack()
    ta_pred_tensor = ta_pred.stack()
    ta_intersect_tensor = ta_intersect.stack()
    IoU_tensor = 2 * ta_intersect_tensor / (ta_true_tensor + ta_pred_tensor + 1e-4)
    IoU_mean = K.mean(IoU_tensor)

    return IoU_mean

def metrics_overall_loop(t_true, t_pred, i, ta_true, ta_pred, ta_intersect):
    ### Get the current iteration's tru and predicted tensor
    c_true = t_true[i]
    c_pred = t_pred[i]

    _, _, intersect_mask_confid = confidence_metric(c_true, c_pred)
    intersect_mask_class = class_metric(c_true, c_pred)
    area_true, area_pred, area_intersect = coord_metric(c_true, c_pred)
    area_intersect_valid = K.sum(intersect_mask_confid * intersect_mask_class * area_intersect)

    ta_true = ta_true.write(i, area_true)
    ta_pred = ta_pred.write(i, area_pred)
    ta_intersect = ta_intersect.write(i, area_intersect_valid)
    return t_true, t_pred, i + 1, ta_true, ta_pred, ta_intersect

def yolo_metric(y_true, y_pred):
    c = lambda t, p, i, ta_true, ta_pred, ta_intersect : K.less(i, K.shape(t)[0])
    ta_true = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    ta_pred = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    ta_intersect = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

    # tf.while_loop creates a Tensorflow map with our loss function calculation (in loop_body())
    t, p, i, ta_true, ta_pred, ta_intersect = tf.while_loop(c, metrics_overall_loop, [y_true, y_pred, 0, ta_true, ta_pred, ta_intersect])

    ### convert TensorArray into a tensor and calculate mean metrics
    ta_true_tensor = ta_true.stack()
    ta_pred_tensor = ta_pred.stack()
    ta_intersect_tensor = ta_intersect.stack()
    IoU_tensor = 2 * ta_intersect_tensor / (ta_true_tensor + ta_pred_tensor + 1e-4)
    IoU_mean = K.mean(IoU_tensor)

    return IoU_mean

def train(modelName, boxlabel_df, trainFiles, validFiles, batch_size = 16, shrink = 1, epoch = 10, 
            continueTraining = False, unlockResnet = False, loadedModel = None):
    traingen = ODTrainingDataGenerator(boxlabel_df, trainFiles, batch_size = batch_size, shrink = shrink, shuffle = True, augmentation = sd_util.AUGMENTATION)
    validgen = ODTrainingDataGenerator(boxlabel_df, validFiles, batch_size = batch_size, shrink = shrink, augmentation = sd_util.AUGMENTATION)
    print("")
    print("Train start at: {}".format(datetime.now()))

    filepathname = os.path.join(sd_util.MODEL_FOLDER, modelName + ".h5")

    if loadedModel is None:
        model = load_model(filepathname, custom_objects = 
            {   "yolo_loss": yolo_loss, 
                "yolo_metric_confid": yolo_metric_confid, 
                "yolo_metric_class": yolo_metric_class,
                "yolo_metric_coord": yolo_metric_coord,
                "yolo_metric": yolo_metric})
    else:
        model = loadedModel

#    metrics = [yolo_metric, yolo_metric_confid, yolo_metric_class, yolo_metric_coord]
    metrics = [yolo_metric_confid, yolo_metric_class, yolo_metric_coord]

    if continueTraining:
        if unlockResnet:
            unfreezeResnet(model)
            model.compile(optimizer = "adam", loss = yolo_loss, metrics = metrics)
    else:
        model.compile(optimizer = "adam", loss = yolo_loss, metrics = metrics)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_yolo_metric_confid', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

#    sess = K.get_session()
#    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#    K.set_session(sess)

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

    return model, newModelName

if __name__ == "__main__":
    boxlabel_df = sd_util.loadBoxLabels()
#    test_ODTrainingDataGenerator()
    trainFiles, validFiles = splitTrainingDataSet(os.path.join(sd_util.PREPROCESSED_FOLDER, "LabeledFiles.txt"))

    # This two line for fast run training to verify code logic.
    if sd_util.FAST_VERIFICATION:
        trainFiles = trainFiles[:2000]
        validFiles = validFiles[:300]

    modelName = "YOLO_resnet34_800_128_1_original"
    model, modelName = train(sd_util.MODEL_FOLDER, modelName, boxlabel_df, trainFiles, validFiles, batch_size = 8, epoch = 5, shrink = 2, continueTraining = True, unlockResnet = True)
    model, modelName = train(sd_util.MODEL_FOLDER, modelName, boxlabel_df, trainFiles, validFiles, batch_size = 8, epoch = 5, shrink = 2, continueTraining = True, unlockResnet = False)
    sd_util.AUGMENTATION = True
    model, modelName = train(sd_util.MODEL_FOLDER, modelName, boxlabel_df, trainFiles, validFiles, batch_size = 8, epoch = 5, shrink = 2, continueTraining = True, unlockResnet = False, loadedModel = model)
