import numpy as np
import pandas as pd
from . import util as sd_util
import os
import sys
from os import listdir
from os.path import isfile, join
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused 
from matplotlib import pyplot as plt

def getAverageBrightness():
    trainImageFiles = [f for f in listdir(sd_util.TRAIN_IMAGES) if isfile(join(sd_util.TRAIN_IMAGES, f))]
    fileIndexes = np.arange(len(trainImageFiles))
    np.random.shuffle(fileIndexes)
    sum = 0
    for i in range(1000):
        image = sd_util.loadImage(os.path.join(sd_util.TRAIN_IMAGES, trainImageFiles[fileIndexes[i]]))
        sum += np.sum(image)
        if i == 0 or i == 9 or i == 99 or i == 999:
            average = sum * 1.0 / (sd_util.IMAGE_HEIGHT * sd_util.IMAGE_WIDTH * (i + 1))
            print("Average for {} files: {}".format(i + 1, average))


# In this method, we will collect these informations:
# 1. Files that have labels;
# 2. Files of each class that;
# 3. A long list of labels with format [class, width, height, maskArea]
def statisticsLabels(label_df):
    filesWithLabels = set()
    FilesOfClasses = [set(), set(), set(), set()]
    classes = []
    widths = []
    heights = []
    maskAreas = []

    totalFileCount = int(label_df.shape[0] / 4)

    for index, row in label_df.iterrows():
        if row['hasRle']:
            if row["ImageId"] not in filesWithLabels:
                filesWithLabels.add(row["ImageId"])
            FilesOfClasses[row["ClassId"] - 1].add(row["ImageId"])
            rects, maskedAreas = sd_util.rle2rects(row['EncodedPixels'])
            for i in range(len(rects)):
                classes.append(row["ClassId"])
                widths.append(rects[i][2] - rects[i][0])
                heights.append(rects[i][3] - rects[i][1])
                maskAreas.append(maskedAreas[i])

            print("{}\r".format(index), end = "", file=sys.stderr)

    labeledFileCount = len(filesWithLabels)
    print("\n{} files with labels.".format(labeledFileCount))
    for f in filesWithLabels:
        print(f)

    for i in range(4):
        print("\n{} files with class {} labels.".format(len(FilesOfClasses[i]), i + 1))
        for f in FilesOfClasses[i]:
            print(f)

    details = np.zeros((len(classes), 4))
    details[:, 0] = np.array(classes).T
    details[:, 1] = np.array(widths).T
    details[:, 2] = np.array(heights).T
    details[:, 3] = np.array(maskAreas).T

    print("\nThere are {} files, {} files with labels, {} masks.".format(totalFileCount, labeledFileCount, len(classes)))
    sumMaskArea = np.sum(details[:, 3])
    imageArea = sd_util.IMAGE_HEIGHT * sd_util.IMAGE_WIDTH * 1.0
    print("Mask ratio {} to all files, {} to labels files".format(sumMaskArea / (imageArea * totalFileCount), sumMaskArea / (imageArea * labeledFileCount)))

    np.save("details.npy", details)

def plotScatter(class_pd):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(class_pd["HWRatio"], class_pd["MaskRatio"], class_pd["MaskArea"])

    ax.set_xlabel('HWRatio')
    ax.set_ylabel('MaskRatio')
    ax.set_zlabel('MaskArea')

    plt.show()

def analyzeInClass(class_pd):
    print("For Class {}".format(int(class_pd.iloc[0].ClassId)))
    print("Total mask Count: {}".format(int(class_pd["MaskArea"].count())))
    print("Total mask Area: {}".format(int(class_pd["MaskArea"].sum())))
    plotScatter(class_pd)

def alalyzeMasks():
    details = np.load("details.npy")
    # Analyze distribution of masks

    # The structure of detais is [class, width, height, maskArea]
    # We will convert it to [class, H/W ratio, mask ratio, maskArea]
    smooth = 1
    areaRatio = (details[:, 3] + smooth)/ ((details[:, 1] * details[:, 2]) + smooth)
    details[:, 1] = (details[:, 2] + smooth) / (details[:, 1] + smooth)
    details[:, 2] = areaRatio

    mask_pd = pd.DataFrame({'ClassId': details[:, 0], 'HWRatio': details[:, 1], 'MaskRatio': details[:, 2], 'MaskArea': details[:, 3]})
    # Remove all those MaskRatio > 1 rows.
    mask_pd = mask_pd[mask_pd.MaskRatio <= 1]
    mask_pd.to_csv("details.csv", index = False)
    # Also remove those HWRatio > 100 rows
    mask_pd = mask_pd[mask_pd.HWRatio <= 100]

    print("Total Mask Area: {}".format(mask_pd["MaskArea"].sum()))
    for cls in range(4):
        classId = cls + 1
        class_pd = mask_pd[mask_pd.ClassId == classId]
        analyzeInClass(class_pd)

def shuffleFiles(path):
    files = pd.read_csv(path)
    np.random.shuffle(files.values)
    files.to_csv(path, index = False)

# The naming of the extract small picture is: XX_XXX_ImageId.jpg
# The first XX is the H/W ratio times 10
# The second XXX is rect index in this rle. This index may change as 
# long as the impl of rle2rects changed.
def extractMaskedRectFromClass(label_df, savePath, targetClassIds):
    rowCount = label_df.shape[0]
    boxLabels = []
    for index, row in label_df.iterrows():
        classId = row["ClassId"]
        if classId in targetClassIds:
            imageid_classid =  row["ImageId_ClassId"]
            if row["hasRle"] == True:
                print("{}/{}/{}     \r".format(classId, index, rowCount), end = "", file=sys.stderr)
                rle = row["EncodedPixels"]
                rects, _ = sd_util.rle2rects(rle)
                img = sd_util.loadImage(os.path.join(sd_util.TRAIN_IMAGES, row["ImageId"]))
                mask = sd_util.rle2mask(rle) * 128
                rectStr = ""
                for i in range(len(rects)):
                    rectStr += " {} {} {} {}".format(rects[i][0], rects[i][1], rects[i][2], rects[i][3])
                    if (rects[i][3] - rects[i][1] + 1) * (rects[i][2] - rects[i][0] + 1) <= 4:
                        continue # We do not care such small point
                    subregion = img[rects[i][1] : rects[i][3] + 1, rects[i][0] : rects[i][2] + 1]
                    # Create file path
                    hwratio = int(10.0 * (rects[i][3] + 1 - rects[i][1]) / (rects[i][2] + 1 - rects[i][0]))
                    if hwratio > 99:
                        hwratio = 99
                    fileName = "{:02d}_{:01d}_{:03d}_{}".format(hwratio, classId, i, row["ImageId"])
                    filePath = os.path.join(sd_util.DATA_DIR, savePath, fileName)
                    sd_util.saveImage(filePath, subregion)

                    # Also save mask into icons
                    subregion = mask[rects[i][1] : rects[i][3] + 1, rects[i][0] : rects[i][2] + 1]
                    fileName = "M_{}".format(fileName)
                    filePath = os.path.join(sd_util.DATA_DIR, savePath, fileName)
                    sd_util.saveImage(filePath, subregion)
                boxLabels.append("{},{}".format(imageid_classid, rectStr[1:]))    
            else:
                boxLabels.append("{},".format(imageid_classid))    

    # Save the box labels to file
    longString = '\r'.join(str(x) for x in boxLabels)
    filePath = os.path.join(sd_util.DATA_DIR, savePath, "BoxLabels.csv")
    fh = open(filePath, "w")
    fh.write(longString)
    fh.close()

# Test code
if __name__ == "__main__":
    print(os.getcwd())
#    getAverageBrightness()

    label_df = sd_util.loadLabels()
    extractMaskedRectFromClass(label_df, "featureImages2", set([1, 2, 3, 4]))
#    statisticsLabels(label_df)

#    alalyzeMasks()
#    shuffleFiles("AllFiles.txt")
#    shuffleFiles("LabeledFiles.txt")
#    shuffleFiles("Class1Files.txt")
#    shuffleFiles("Class2Files.txt")
#    shuffleFiles("Class3Files.txt")
#    shuffleFiles("Class4Files.txt")

