import keras
import numpy as np
from keras import backend as K
import tensorflow as tf

class DiceCoef:
    def __init__(self, threshold = 0.5):
        self.threshold = threshold
        self.__name__ = "DiceCoef"

    def __call__(self, y_true, y_pred, smooth = 1):
        y_pred = K.cast(K.greater(y_pred, self.threshold), 'float32')
        intersection = y_true * y_pred
        score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
        return score

# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss

class WeightedBCELoss:
    def __init__(self, weight = 0.6):
        self.weight = weight
        self.__name__ = "WeightedBCELoss"

    def __call__(self, y_true, y_pred):
        y_pred = K.tensorflow_backend.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_pred = K.tensorflow_backend.log(y_pred / (1 - y_pred))
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight = self.weight)

def iouLoss(y_true, y_pred, smooth = 1e-6):
    intersection = K.sum(y_true * y_pred)
    total = K.sum(y_true) + K.sum(y_pred)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

# The Dice coefficient, or Sørensen–Dice coefficient, is a common metric for binary 
# classification tasks such as pixel segmentation that can also be modified to act 
# as a loss function
def diceLoss(y_true, y_pred, smooth = 1):
    intersection = K.sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

# This loss combines Dice loss with the standard binary cross-entropy (BCE) loss that 
# some diversity in the loss, while benefitting from the stability of BCE. The equation 
# for multi-class BCE by itself will be familiar to anyone who has studied logistic 
# regression.
def diceBCELoss(y_true, y_pred, smooth = 1):    
    return K.binary_crossentropy(y_true, y_pred) + diceLoss(y_true, y_pred, smooth)

# This loss was introduced in "Tversky loss function for image segmentationusing 3D 
# fully convolutional deep networks", retrievable here: https://arxiv.org/abs/1706.05721. 
# It was designed to optimise segmentation on imbalanced medical datasets by utilising 
# constants that can adjust how harshly different types of error are penalised in the 
# loss function. From the paper:
#
#   in the case of α=β=0.5 the Tversky index simplifies to be the same as the Dice 
#   coefficient, which is also equal to the F1 score. With α=β=1, Equation 2 produces 
#   Tanimoto coefficient, and setting α+β=1 produces the set of Fβ scores. Larger βs 
#   weigh recall higher than precision (by placing more emphasis on false negatives).
#
# To summarise, this loss function is weighted by the constants 'alpha' and 'beta' that 
# penalise false positives and false negatives respectively to a higher degree in the 
# loss function as their value is increased. The beta constant in particular has 
# applications in situations where models can obtain misleadingly positive performance 
# via highly conservative prediction. You may want to experiment with different values 
# to find the optimum. With alpha==beta==0.5, this loss becomes equivalent to Dice Loss.
class TverskyLoss:
    def __init__(self, alpha = 0.3, beta = 0.7):
        self.alpha = alpha
        self.beta = beta
        self.__name__ = "TverskyLoss"

    def __call__(self, y_true, y_pred, smooth = 1):
        TP = K.sum(y_true * y_pred)
        FP = K.sum((1 - y_true) * y_pred)
        FN = K.sum(y_true * (1 - y_pred))
        tversky_coef = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        return 1 - tversky_coef

# Focal Loss was introduced by Lin et al of Facebook AI Research in 2017 as a means of 
# combatting extremely imbalanced datasets where positive cases were relatively rare. 
# Their paper "Focal Loss for Dense Object Detection" is retrievable here: 
# https://arxiv.org/abs/1708.02002. In practice, the researchers used an alpha-modified 
# version of the function so I have included it in this implementation.
class FocalLoss:
    def __init__(self, alpha = 0.8, gamma = 2.0):
        self.alpha = alpha
        self.gamma = gamma
        self.__name__ = "FocalLoss"

    def __call__(self, y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        bce_exp = K.exp(-bce)
        return K.mean(self.alpha * K.pow((1 - bce_exp), self.gamma) * bce)

# Test code
if __name__ == "__main__":
    diceCoef = DiceCoef()
    weightedBCELoss = WeightedBCELoss()
    tverskyLoss = TverskyLoss()
    focalLoss = FocalLoss()

    y_true = np.array([[1, 2], [3, 4]])
    y_pred = np.array([[5, 6], [7, 8]])
    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)

    print(iouLoss(y_true, y_pred))
    print(diceLoss(y_true, y_pred))
    print(diceBCELoss(y_true, y_pred))

    print(diceCoef(y_true, y_pred))
    print(weightedBCELoss(y_true, y_pred))
    print(tverskyLoss(y_true, y_pred))
    print(focalLoss(y_true, y_pred))

    pass