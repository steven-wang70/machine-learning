import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

DATA_DIR = None
TRAIN_IMAGES = None
TEST_IMAGES = None
TRAIN_LABELS_FILE = None
MODEL_FOLDER = None
PREPROCESSED_FOLDER = None

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 1600
NUM_CLASS = 4
VALID_SET_RATIO = 0.3
FAST_VERIFICATION = False
AUGMENTATION = True
SHOW_PROGRESS = True

def initContext(dataFolder, modelFolder, preprocessedFolder):
    global DATA_DIR, TRAIN_IMAGES, TEST_IMAGES, MODEL_FOLDER, TRAIN_LABELS_FILE, PREPROCESSED_FOLDER
    DATA_DIR = dataFolder
    TRAIN_IMAGES = os.path.join(DATA_DIR, "train_images")
    TEST_IMAGES = os.path.join(DATA_DIR, "test_images")
    TRAIN_LABELS_FILE = os.path.join(DATA_DIR, "train.csv")
    MODEL_FOLDER = modelFolder
    PREPROCESSED_FOLDER = preprocessedFolder

def loadImage(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def saveImage(path, data):
    cv2.imwrite(path, data)

def resizeImage(originImg, newHeight, newWidth):
    newImage = cv2.resize(originImg, (newWidth, newHeight))
    return newImage

# Return a minimum rectangle that could hold all these points.
def points2rect(points):
    leftX = np.min(points[:, 0])
    rightX = np.max(points[:, 0])
    topY = np.min(points[:, 1])
    bottomY = np.max(points[:, 1])

    return (leftX, topY, rightX, bottomY)

def getRleArea(rle):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    return np.sum(lengths)

def encoded2rects(encoded):
    s = encoded.split()
    points = np.asarray(s, dtype = int)
    rects = points.reshape(points.shape[0] // 4, 4)
    return rects

# Convert rles encoded masks to its close rectangles.
def rle2rects(rle, imageSize = (IMAGE_HEIGHT, IMAGE_WIDTH)):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    # From the observaton, some run-length may cover multiple lines.
    rlesInSingleLine = [] # [[X, top, bottom]...]
    for rl in range(starts.shape[0]):
        startc = starts[rl] // imageSize[0]
        startr = starts[rl] % imageSize[0]
        endr = startr + lengths[rl]
        col = [startc, startr, endr]
        rlesInSingleLine.append(col)
        # If this is a multiple lines run length
        if col[2] > IMAGE_HEIGHT:
            remaining = col[2] - IMAGE_HEIGHT
            col[2] = IMAGE_HEIGHT
            nextColIndex = col[0] + 1
            while remaining > 0:
                thisColLen = remaining
                if thisColLen >= IMAGE_HEIGHT:
                    thisColLen = IMAGE_HEIGHT
                rlesInSingleLine.append([nextColIndex, 0, thisColLen])
                remaining -= IMAGE_HEIGHT
                nextColIndex += 1

    rles = np.array(rlesInSingleLine)

    # Grouping all these rles
    groups = [] # for each item, the encoding is [lineIndex1, lineIndex2, ...]
    for index in range(rles.shape[0]):
        groupFound = None
        for gIndex in range(len(groups)):
            for lineInGroup in groups[gIndex]:
                if  rles[index][0] == rles[lineInGroup][0] + 1 and \
                    rles[index][1] < rles[lineInGroup][2] and \
                    rles[index][2] > rles[lineInGroup][1]: # They are adjacent.
                    if groupFound is None:
                        groups[gIndex].append(index)
                        groupFound = gIndex # We found a group for it
                    else: # This line is connecting two groups which causes a merge.
                        groups[groupFound] += groups[gIndex]
                        groups[gIndex].clear()
                    break

        if groupFound is None:
            groups.append([index])

    rects = []
    maskedAreas = []
    for g in groups:
        if len(g) == 0:
            continue
        # Convert group to a numpy array of [?, 2], which is the [x, y] of each line end
        endPoints = np.zeros((len(g) * 2, 2), dtype=int)
        area = 0
        for i in range(len(g)):
            endPoints[2 * i, 0] = rles[g[i]][0]
            endPoints[2 * i, 1] = rles[g[i]][1]
            endPoints[2 * i + 1, 0] = rles[g[i]][0]
            endPoints[2 * i + 1, 1] = rles[g[i]][2]
            area += rles[g[i]][2] - rles[g[i]][1]
        rects.append(points2rect(endPoints))
        maskedAreas.append(area)

    return rects, maskedAreas

def test_rle2rects(label_df):
    imageId = "047fa15d0.jpg"
    classId = 3
    rle = getRleFromLabel(label_df, imageId, classId)
    rects, maskedAreas = rle2rects(rle)
    imgpath = os.path.join(TRAIN_IMAGES, imageId)
    for i in range(len(rects)):
        rectArea = ((rects[i][2] - rects[i][0])) * ((rects[i][3] - rects[i][1]))
        print("Ratio: {}, {}".format(maskedAreas[i], maskedAreas[i] * 1.0 / rectArea))
    showImageWithRects(imgpath, rects)

def rle2mask(rle, shape = (IMAGE_HEIGHT, IMAGE_WIDTH)):
    if rle is None or rle == "":
        return np.zeros(shape ,dtype=np.uint8)

    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1

    mask= np.zeros( shape[0] * shape[1] ,dtype=np.uint8)
    for index, start in enumerate(starts):
        mask[start : start + lengths[index]] = 1
    
    return mask.reshape(shape, order = 'F')

def mask2rle(mask):
    mask= mask.T.flatten()
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
    
def expandMask(mask, delta):
    w = mask.shape[1]
    h = mask.shape[0]
    
    # MASK UP
    for k in range(1,delta,2):
        temp = np.concatenate([mask[k:,:],np.zeros((k,w))],axis=0)
        mask = np.logical_or(mask,temp)
    # MASK DOWN
    for k in range(1,delta,2):
        temp = np.concatenate([np.zeros((k,w)),mask[:-k,:]],axis=0)
        mask = np.logical_or(mask,temp)
    # MASK LEFT
    for k in range(1,delta,2):
        temp = np.concatenate([mask[:,k:],np.zeros((h,k))],axis=1)
        mask = np.logical_or(mask,temp)
    # MASK RIGHT
    for k in range(1,delta,2):
        temp = np.concatenate([np.zeros((h,k)),mask[:,:-k]],axis=1)
        mask = np.logical_or(mask,temp)
    
    return mask 

def mask2contour(mask, width):
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3) 

def test_mask(label_df):
    imageId = "047fa15d0.jpg"
    classId = 3
    rle = getRleFromLabel(label_df, imageId, classId)
    mask = rle2mask(rle)
    imgpath = os.path.join(TRAIN_IMAGES, imageId)
    showImageWithMask(imgpath, mask)
    contour = mask2contour(mask, 3)
    showImageWithMask(imgpath, contour)
    expandMask(mask, 5)
    showImageWithMask(imgpath, mask)

    rle2 = mask2rle(mask)
    if rle2 == rle:
        print("mask2rle works.")
    else:
        print("mask2rle failed.")
        print(rle)
        print(rle2)

def loadLabels(path):
    label_df = pd.read_csv(path)
    label_df['ImageId'] = label_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    label_df['ClassId'] = label_df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
    label_df['hasRle'] = ~ label_df['EncodedPixels'].isna()

    return label_df

def loadBoxLabels(path = os.path.join(".", "BoxLabels.csv")):
    label_df = pd.read_csv(path)
    label_df['ImageId'] = label_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    label_df['ClassId'] = label_df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
    label_df['hasBox'] = ~ label_df['EncodedBoxes'].isna()

    return label_df

def getRleFromLabel(label_df, imageId, classId):
    imageId_ClassId = "{}_{}".format(imageId, classId)
    result = label_df[label_df.ImageId_ClassId == imageId_ClassId]
    if result.size > 0:
        if result.iloc[0].hasRle:
            return result.iloc[0].EncodedPixels

    return None
    
def saveTestResult(results, filepath):
    report = open(filepath, "w")
    report.write("ImageId_ClassId,EncodedPixels\n")
    for r in results:
        report.write("{}\n".format(r))
    report.close()

def showImageWithMask(imgpath, mask):
    img = loadImage(imgpath)
    img[mask == 1] = 255

    plt.imshow(img, cmap='gray')
    plt.show()

def showImageWithRects(imgPath, rects):
    img = loadImage(imgPath)
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img, cmap='gray')

    # Create a Rectangle patch
    for r in rects:
        rect = patches.Rectangle((r[0], r[1]), r[2] - r[0],r[3] - r[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def showMatrixWithRects(matrix, rects):
    fig, ax = plt.subplots(1)

    # Display the image
    ax.matshow(matrix, cmap='gray')

    # Create a Rectangle patch
    for r in rects:
        rect = patches.Rectangle((r[0], r[1]), r[2] - r[0],r[3] - r[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def showMatrix(matrix):
    fig, ax = plt.subplots(1)

    # Display the image
    ax.matshow(matrix)
    plt.show()

def pltContours(matrix):
    x = np.arange (matrix.shape[2])
    y = np.arange (matrix.shape[1])
    X, Y = np.meshgrid(x, y)
    row = matrix.shape[0]
    for i in range(row):
        plt.subplot(row, 1, i + 1)
        plt.contour(X, Y, matrix[i, :, :])

    plt.show()

def plotContour(matrix):
    x = np.arange (matrix.shape[1])
    y = np.arange (matrix.shape[0])
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, matrix)

    plt.show()

def pltHistogram(data):
    data = data.flatten()
    num_bins = 100

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(data, num_bins, density=1, log = True)
    
    # add a 'best fit' line
    mean = np.mean(data)
    sigma = np.std(data)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mean))**2))
    ax.plot(bins, y, '--')
       
    fig.tight_layout()
    plt.show()

# Test code
if __name__ == "__main__":
    label_df1 = loadLabels()

    test_mask(label_df1)
    test_rle2rects(label_df1)

