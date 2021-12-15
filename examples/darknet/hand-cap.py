import numpy as np
import os
import urllib.request
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import sys
import math
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import time

GRID0 = 13
GRID1 = 26
GRID2 = 52
LISTSIZE = 6
SPAN = 3
NUM_CLS = 1
MAX_BOXES = 500
OBJ_THRESH = 0.5
NMS_THRESH = 0.6


CLASS = "Hand"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    box_wh = box_wh * anchors

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def hand_post_process(input_data):
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
            [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nscores.append(s[keep])

    if not nscores:
        return None, None

    boxes = np.concatenate(nboxes)
    scores = np.concatenate(nscores)

    return boxes, scores

def draw(image, boxes, scores):
    for box, score in zip(boxes, scores):
        x, y, w, h = box
        print('class: {}, score: {}'.format(CLASS, score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv.putText(image, '{0} {1:.2f}'.format(CLASS, score),
                    (top, left - 6),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--library", help="Path to C static library file")
    parser.add_argument("--model", help="Path to nbg file")
    parser.add_argument("--device", help="the number for video device")
    parser.add_argument("--level", help="Information printer level: 0/1/2")

    args = parser.parse_args()
    
    if args.model :
        if os.path.exists(args.model) == False:
            sys.exit('Model \'{}\' not exist'.format(args.model))
        model = args.model
    else:
        sys.exit("NBG file not found !!! Please use format: --model")
    if args.device :
        cap_num = args.device
    else :
        sys.exit("video device not found !!! Please use format :--device ")
    if args.library :
        if os.path.exists(args.library) == False :
            sys.exit('C static library \'{}\' not exist'.format(args.library))
        library = args.library
    else :
        sys.exit("C static library not found !!! Please use format: --library")
    if args.level == '1' or args.level == '2' :
        level = int(args.level)
    else :
        level = 0

    hand = KSNN('VIM3')
    print(' |---+ KSNN Version: {} +---| '.format(hand.get_nn_version()))

    print('Start init neural network ...')
    hand.nn_init(library=library, model=model, level=level)
    print('Done.')

    cap = cv.VideoCapture(int(cap_num))
    cap.set(3,1920)
    cap.set(4,1080)
    while(1):
        cv_img = list()
        ret,img = cap.read()
        cv_img.append(img)
        start = time.time()
        '''
               default input_tensor is 1
        '''
        data = hand.nn_inference(cv_img, platform='DARKNET', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)
        end = time.time()
        print('inference : ', end - start)
        input0_data = data[0]
        input1_data = data[1]
        input2_data = data[2]

        input0_data = input0_data.reshape(SPAN, LISTSIZE, GRID0, GRID0)
        input1_data = input1_data.reshape(SPAN, LISTSIZE, GRID1, GRID1)
        input2_data = input2_data.reshape(SPAN, LISTSIZE, GRID2, GRID2)

        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

        boxes, scores = hand_post_process(input_data)

        if boxes is not None:
            draw(img, boxes, scores)

        cv.imshow("capture", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows() 
