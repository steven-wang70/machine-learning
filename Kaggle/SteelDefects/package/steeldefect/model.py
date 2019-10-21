from . import util as sd_util
import keras
from keras import regularizers
from keras.layers import BatchNormalization as BatchNorm
from keras.models import Model, load_model
from keras.layers import Input, Reshape, Activation, Flatten, Dense, Dropout, Concatenate
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import layers
from segmentation_models import Unet
from segmentation_models.utils import set_trainable


def buildUNetModel(pathToSave, width, height, outputChannelCount, backbone = 'resnet50'):
    base_model = Unet(backbone, input_shape=(width, height, 3), classes = outputChannelCount, activation='sigmoid', freeze_encoder=True)
    base_model.summary()
    input = Input(shape=(None, None, 1))
    # This adapter layer map grayscale to RGB space since all these public models 3 channels only.
    adapter = Conv2D(3, (1, 1), trainable = False, name = "AdapterLayer", kernel_initializer = "ones")(input)
    out = base_model(adapter)

    model = Model(input, out, name=base_model.name)

    model.save(pathToSave)
    return model

def buildYoloModel(pathToSave, width, height):
    base_model = Unet('resnet34', input_shape=(width, height, 3), classes = 1, activation='sigmoid', freeze_encoder=True)
    # UNet is based on resnet34 in this case, so we could strip those UNet specific layers to get the resnet34 model.
    # The topmost layer of the resnet is "relu1".
    relu1_layer_name = "relu1"
    while base_model.layers[-1].name != relu1_layer_name:
        base_model.layers.pop()
    relu1_layer = base_model.layers[-1]
    # Use this line could create a resnet34 model
    # model = Model(inputs = base_model.input, outputs = relu1_layer)

    X = Flatten(name = "yolo1")(relu1_layer.output)
    X = Dense(2048, activation = "relu", name = "yolo2")(X)
    X = Dense(1024, activation = "relu", name = "yolo3")(X)
    # We are using 25 x 4 grid, each grid has a box of 4 classes.
    # Each box has a confidence, central point, height and width.
    # So in this case, this layers has 900 nodes.
    confidences = Dense(25 * 4, activation = "sigmoid", name = "yolo_conf1")(X) # 0..1
    classes = Dense(25 * 4 * 4, activation = "softmax", name = "yolo_classes")(X) # 0..1
    coord = Dense(25 * 4 * 2, activation = "tanh", name = "yolo_coord1")(X)
    coord = Lambda(lambda x: x / 2, name = "yolo_coord2")(coord) # -0.5..0.5
    sizes = Dense(25 * 4 * 2, activation = "relu", name = "yolo_sizes")(X) # >= 0
    pred = Concatenate(name = "yolo4")([confidences, classes, coord, sizes])
    model = Model(inputs = base_model.input, outputs = pred)
    model.summary()

    # At last, add the first layer adapter
    input = Input(shape=(height, width, 1))
    # This adapter layer map grayscale to RGB space since all these public models 3 channels only.
    adapter = Conv2D(3, (1, 1), trainable = False, name = "AdapterLayer", kernel_initializer = "ones")(input)
    out = model(adapter)

    model = Model(input, out, name=base_model.name)
    model.summary()

    model.save(pathToSave)
    return model


# Test code
if __name__ == "__main__":
    buildUNetModel('UNET_resnext50_800_128_4_original.h5', 800, 128, 4, backbone = "resnext50")
#    buildYoloModel('YOLO_resnet34_800_128_1_original.h5', 800, 128)
