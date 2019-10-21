# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This code implements a Feed forward neural network using Keras API."""

import argparse
import glob
import os
import keras
from steeldefect import util as sd_util
from steeldefect import training as sd_training
from steeldefect import losses as sd_losses

from tensorflow.python.lib.io import file_io
from zipfile import ZipFile

def copyFile(src_file, dest_dir):
  # file_io.copy does not copy files into folders directly.
  file_name = os.path.basename(src_file)
  new_file_location = os.path.join(dest_dir, file_name)
  print("Copy file: {}".format(src_file))
  file_io.copy(src_file, new_file_location)

def copyDir(srcDir, dstDir):
  for dir_name, sub_dirs, leaf_files in file_io.walk(srcDir):
    # copy all the files over
    for leaf_file in leaf_files:
      leaf_file_path = os.path.join(dir_name, leaf_file)
      copyFile(leaf_file_path, dstDir)

    # Now make all the folders.
    for sub_dir in sub_dirs:
      dstSubDir = os.path.join(dstDir, sub_dir)
      file_io.create_dir(dstSubDir)
      copyDir(os.path.join(srcDir, sub_dir), dstSubDir)

# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def train_and_evaluate(args):
  print(args)
  try:
    os.makedirs(args.job_dir)
  except:
    pass

  data_dir = args.data_dir
  preprocessed_dir = args.preprocessed_dir
  if args.data_dir.startswith('gs://'): # OpenCV does not support Google storage. We need to copy them to local
    tempDataDir = "temp_data_dir"
    file_io.create_dir(tempDataDir)
    copyDir(args.data_dir, tempDataDir)

    # Unzip the file train_images.zip
    print("Extract zip file: train_images.zip")
    with ZipFile(os.path.join(tempDataDir, "train_images.zip"), 'r') as zipObj:
      # Extract all the contents of zip file in different directory
      zipObj.extractall(os.path.join(tempDataDir, "train_images"))
    print("Extract zip file completed: train_images.zip")

    data_dir = tempDataDir
    preprocessed_dir = os.path.join(data_dir, "preprocessed")

  model_dir = args.model_dir
  if args.model_dir.startswith('gs://'): # H5py does not support Google storage. We need to copy them to local
    tempModelDir = "temp_model_dir"
    file_io.create_dir(tempModelDir)
    copyFile(os.path.join(args.model_dir, args.model_name + ".h5"), tempModelDir)
    model_dir = tempModelDir

  sd_util.initContext(data_dir, model_dir, preprocessed_dir)
  sd_util.FAST_VERIFICATION = args.fast_verification
  sd_util.SHOW_PROGRESS = args.show_progress

  bigLoop = 1
  epochs = args.num_epochs
  if epochs > 3:
    bigLoop = int(epochs // 3 + 1)
    epochs = 3

  label_df = sd_util.loadLabels(os.path.join(sd_util.DATA_DIR, "train.csv"))
  trainFiles, validFiles = sd_training.splitTrainingDataSet(os.path.join(sd_util.PREPROCESSED_FOLDER, "AllFiles.txt"))

  # This two line for fast run training to verify code logic.
  if sd_util.FAST_VERIFICATION:
    trainFiles = trainFiles[:30]
    validFiles = validFiles[:30]
  print("Train files: {}".format(trainFiles.shape[0]))
  print("Valid files: {}".format(validFiles.shape[0]))

  optimizer = keras.optimizers.Adam()
  continueTraining = args.continue_training
  for _ in range(bigLoop):
    print(args)
    _, modelName = sd_training.train(args.model_name, label_df, trainFiles, validFiles, shrink = 2, 
                         batch_size = args.batch_size, epoch = epochs, lossfunc = sd_losses.diceBCELoss, 
                         optimizer = optimizer, continueTraining = continueTraining, 
                         unlockResnet = args.unlock_backbone)
    continueTraining = True

    if args.model_dir.startswith('gs://'):
      modelFileName = modelName + ".h5"
      copyFile(os.path.join(model_dir, modelFileName), args.model_dir)
      print("Copied H5 file to: {}".format(os.path.join(args.model_dir, modelFileName)))

def str2bool(value):
    return value.lower == 'true'

def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--job-dir',
      type=str,
      help='GCS or local dir to write checkpoints and export model',
      default='')
  parser.add_argument(
      '--batch-size',
      type=int,
      default=40,
      help='Batch size for both training and validation steps')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.003,
      help='Learning rate for optimizers')
  parser.add_argument(
      '--num-epochs',
      type=int,
      default=20,
      help='Maximum number of epochs on which to train')
  parser.add_argument(
      '--preprocessed-dir',
      type=str,
      default="",
      help='The dir of preprocessed files')
  parser.add_argument(
      '--model-dir',
      type=str,
      default="",
      help='The dir of the loaded model')
  parser.add_argument(
      '--model-name',
      type=str,
      default="",
      help='The name of the loaded model')
  parser.add_argument(
      '--data-dir',
      type=str,
      default="",
      help='The path of training samples')
  parser.add_argument(
      '--continue-training',
      type=str2bool,
      default=False,
      help='Whether this is a continue training')
  parser.add_argument(
      '--fast-verification',
      type=str2bool,
      default=False,
      help='Fast verify the code logic')
  parser.add_argument(
      '--unlock-backbone',
      type=str,
      default=None,
      help='The backbone to be unlocked')
  parser.add_argument(
      '--show-progress',
      type=str2bool,
      default=False,
      help='Show the progress counter')

  args, _ = parser.parse_known_args()
  return args

if __name__ == '__main__':
  args = parseArgs()
  train_and_evaluate(args)
