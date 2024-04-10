import numpy as np
import os      
import argparse
import sys
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import re
import math
import random
import time

INPUT_SIZE = 300

NUM_RESULTS = 1917
NUM_CLASSES = 91

Y_SCALE = 10.0
X_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

mean = [127.5, 127.5, 127.5]
var = [128]

CLASSES = ("???","person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","???","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","???","backpack","umbrella","???","???","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","???","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","???","diningtable","???","???","toilet ","???","tvmonitor","laptop  ","mouse    ","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","???","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")



def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1):
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    i = w * h
    u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i

    if u <= 0.0:
        return 0.0

    return i / u


def load_box_priors():
    box_priors_ = []
    fp = open('./box_priors.txt', 'r')
    ls = fp.readlines()
    for s in ls:
        aList = re.findall('([-+]?\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', s)
        for ss in aList:
            aNum = float((ss[0]+ss[2]))
            box_priors_.append(aNum)
    fp.close()

    box_priors = np.array(box_priors_)
    box_priors = box_priors.reshape(4, NUM_RESULTS)

    return box_priors

def calc_position(vaildCnt, candidateBox, predictions, box_priors):
    for i in range(0, vaildCnt):
        if candidateBox[0][i] == -1:
            continue

        n = candidateBox[0][i]
        ycenter = predictions[0][n][0] / Y_SCALE * box_priors[2][n] + box_priors[0][n]
        xcenter = predictions[0][n][1] / X_SCALE * box_priors[3][n] + box_priors[1][n]
        h = math.exp(predictions[0][n][2] / H_SCALE) * box_priors[2][n]
        w = math.exp(predictions[0][n][3] / W_SCALE) * box_priors[3][n]

        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.

        predictions[0][n][0] = ymin
        predictions[0][n][1] = xmin
        predictions[0][n][2] = ymax
        predictions[0][n][3] = xmax


def nms(vaildCnt, candidateBox, predictions):
    for i in range(0, vaildCnt):
        if candidateBox[0][i] == -1:
            continue

        n = candidateBox[0][i]
        xmin0 = predictions[0][n][1]
        ymin0 = predictions[0][n][0]
        xmax0 = predictions[0][n][3]
        ymax0 = predictions[0][n][2]

        for j in range(i+1, vaildCnt):
            m = candidateBox[0][j]

            if m == -1:
                continue

            xmin1 = predictions[0][m][1]
            ymin1 = predictions[0][m][0]
            xmax1 = predictions[0][m][3]
            ymax1 = predictions[0][m][2]

            iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1)

            if iou >= 0.45:
                candidateBox[0][j] = -1


def draw(img, vaildCnt, candidateBox, predictions, scoreBox):
    for i in range(0, vaildCnt):
        if candidateBox[0][i] == -1:
            continue

        n = candidateBox[0][i]

        xmin = int(max(0.0, min(1.0, predictions[0][n][1])) * img.shape[1])
        ymin = int(max(0.0, min(1.0, predictions[0][n][0])) * img.shape[0])
        xmax = int(max(0.0, min(1.0, predictions[0][n][3])) * img.shape[1])
        ymax = int(max(0.0, min(1.0, predictions[0][n][2])) * img.shape[0])

        print("%d @ (%d, %d) (%d, %d) score=%f" % (candidateBox[1][i], xmin, ymin, xmax, ymax, scoreBox[0][i]))
        cv.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv.putText(img, '{0} {1:.2f}'.format(CLASSES[candidateBox[1][i]], scoreBox[0][i]),
                    (xmin, ymin),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    cv.imshow("results", img)
    cv.waitKey(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--library", help="Path to C static library file")
    parser.add_argument("--model", help="Path to nbg file")
    parser.add_argument("--picture", help="Path to input picture")
    parser.add_argument("--level", help="Information printer level: 0/1/2")
 
    args = parser.parse_args()
    if args.model :
        if os.path.exists(args.model) == False:
            sys.exit('Model \'{}\' not exist'.format(args.model))
        model = args.model
    else :
        sys.exit("NBG file not found !!! Please use format: --model")
    if args.picture :
        if os.path.exists(args.picture) == False:
            sys.exit('Input picture \'{}\' not exist'.format(args.picture))
        picture = args.picture
    else :
        sys.exit("Input picture not found !!! Please use format: --picture")
    if args.library :
        if os.path.exists(args.library) == False:
            sys.exit('C static library \'{}\' not exist'.format(args.library))
        library = args.library
    else :
        sys.exit("C static library not found !!! Please use format: --library")
    if args.level == '1' or args.level == '2' :
        level = int(args.level)
    else :
        level = 0

    ssd = KSNN('VIM3')
    print(' |---+ KSNN Version: {} +---| '.format(ssd.get_nn_version()))

    print('Start init neural network ...')
    ssd.nn_init(library=library, model=model, level=level)
    print('Done.')

    print('Get input data ...')
    cv_img = []
    orig_img = cv.imread(picture, cv.IMREAD_COLOR)
    img = cv.resize(orig_img, (300, 300)).astype(np.float32)
    img[:, :, 0] = img[:, :, 0] - mean[0]
    img[:, :, 1] = img[:, :, 1] - mean[1]
    img[:, :, 2] = img[:, :, 2] - mean[2]
    img = img / var[0]
    
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    img = img
    cv_img.append(img)
    print('Done,')

    print('Start inference ...')
    start = time.time()
    '''
        default input_tensor is 1
    '''
    outputs = ssd.nn_inference(cv_img,platform = 'TENSORFLOW', output_tensor=2, reorder='0 1 2', output_format=output_format.OUT_FORMAT_FLOAT32)
    end = time.time()
    print('Done. inference : ', end - start)

    predictions = outputs[0].reshape((1, NUM_RESULTS, 4))
    outputClasses = outputs[1].reshape((1, NUM_RESULTS, NUM_CLASSES))
    candidateBox = np.zeros([2, NUM_RESULTS], dtype=int)
    scoreBox = np.zeros([1, NUM_RESULTS], dtype=float)
    vaildCnt = 0

    box_priors = load_box_priors()

    # Post Process
    # got valid candidate box
    for i in range(0, NUM_RESULTS):
        topClassScore = -1000
        topClassScoreIndex = -1

    # Skip the first catch-all class.
        for j in range(1, NUM_CLASSES):
            score = sigmoid(outputClasses[0][i][j]);

            if score > topClassScore:
                topClassScoreIndex = j
                topClassScore = score

        if topClassScore > 0.3:
            candidateBox[0][vaildCnt] = i
            candidateBox[1][vaildCnt] = topClassScoreIndex
            scoreBox[0][vaildCnt] = topClassScore
            vaildCnt += 1

    # calc position
    calc_position(vaildCnt, candidateBox, predictions, box_priors)

    # NMS
    nms(vaildCnt, candidateBox, predictions)

    # Draw result
    draw(orig_img, vaildCnt, candidateBox, predictions, scoreBox)

