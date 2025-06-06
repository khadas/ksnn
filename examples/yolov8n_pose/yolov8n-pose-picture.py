import numpy as np
import os
import urllib.request
#from matplotlib import gridspec
#from matplotlib import pyplot as plt
#from PIL import Image
import argparse
import sys
import math
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import time

GRID0 = 20
GRID1 = 40
GRID2 = 80
LISTSIZE = 116
SPAN = 1
NUM_CLS = 1
MAX_BOXES = 500
OBJ_THRESH = 0.4
NMS_THRESH = 0.5
POINT_THRESH = 0.3
mean = [0, 0, 0]
var = [255]

constant_martix = np.array([[0,  1,  2,  3,
			     4,  5,  6,  7,
			     8,  9,  10, 11,
			     12, 13, 14, 15]]).T

line_points = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [5, 11], [6, 8], [6, 12], [7, 9], [8, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
point_colors = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
line_colors = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
colors = [[0, 255, 0], [255, 0, 0], [0, 0, 255]]

CLASSES = ["person"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=0):
	x = np.exp(x)
	return x / x.sum(axis=axis, keepdims=True)

def process(input):

    grid_h, grid_w = map(int, input.shape[0:2])

    box_class_probs = sigmoid(input[..., :NUM_CLS])
    
    box_0 = softmax(input[..., NUM_CLS: NUM_CLS + 16], -1)
    box_1 = softmax(input[..., NUM_CLS + 16:NUM_CLS + 32], -1)
    box_2 = softmax(input[..., NUM_CLS + 32:NUM_CLS + 48], -1)
    box_3 = softmax(input[..., NUM_CLS + 48:NUM_CLS + 64], -1)
    
    result = np.zeros((grid_h, grid_w, 1, 4))
    result[..., 0] = np.dot(box_0, constant_martix)[..., 0]
    result[..., 1] = np.dot(box_1, constant_martix)[..., 0]
    result[..., 2] = np.dot(box_2, constant_martix)[..., 0]
    result[..., 3] = np.dot(box_3, constant_martix)[..., 0]
    
    key_point_result = input[..., NUM_CLS + 64:]
    key_point_result[..., NUM_CLS + 64 + 2::3] = sigmoid(key_point_result[..., NUM_CLS + 64 + 2::3])

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1)
    row = row.reshape(grid_h, grid_w, 1, 1)
    grid = np.concatenate((col, row), axis=-1)

    result[..., 0:2] = (0.5 - result[..., 0:2] + grid) / (grid_w, grid_h)
    result[..., 2:4] = (0.5 + result[..., 2:4] + grid) / (grid_w, grid_h)
    
    key_point_result[..., 0::3] = (key_point_result[..., 0::3] * 2 + col) / grid_w
    key_point_result[..., 1::3] = (key_point_result[..., 1::3] * 2 + row) / grid_h

    return result, box_class_probs, key_point_result

def filter_boxes(boxes, box_class_probs, key_points):

    box_classes = np.argmax(box_class_probs, axis=-1)
    box_class_scores = np.max(box_class_probs, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]
    points = key_points[pos]

    return boxes, classes, scores, points

def nms_boxes(boxes, scores):

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov3_post_process(input_data):
    boxes, classes, scores, key_points = [], [], [], []
    for i in range(3):
        result, confidence, key_point = process(input_data[i])
        b, c, s, p = filter_boxes(result, confidence, key_point)
        boxes.append(b)
        classes.append(c)
        scores.append(s)
        key_points.append(p)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)
    key_points = np.concatenate(key_points)

    nboxes, nclasses, nscores, npoints = [], [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        p = key_points[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
        npoints.append(p[keep])

    if not nclasses and not nscores:
        return None, None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    key_points = np.concatenate(npoints)

    return boxes, scores, classes, key_points

def draw(image, boxes, scores, classes, key_points):

    for box, score, cl, kp in zip(boxes, scores, classes, key_points):
        x1, y1, x2, y2 = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x1, y1, x2, y2))
        x1 *= image.shape[1]
        y1 *= image.shape[0]
        x2 *= image.shape[1]
        y2 *= image.shape[0]
        left = max(0, np.floor(x1 + 0.5).astype(int))
        top = max(0, np.floor(y1 + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x2 + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y2 + 0.5).astype(int))

        cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        #cv.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
        #            (left, top - 6),
        #            cv.FONT_HERSHEY_SIMPLEX,
        #            0.6, (0, 0, 255), 2)
        
        for i in range(17):
            if kp[i * 3 + 2] >= POINT_THRESH:
                point_x = round(kp[i * 3] * image.shape[1])
                point_y = round(kp[i * 3 + 1] * image.shape[0])
                cv.circle(image, (point_x, point_y), 2, colors[point_colors[i]], 2)
        
        for i in range(19):
            if kp[line_points[i][0] * 3 + 2] >= POINT_THRESH and kp[line_points[i][1] * 3 + 2] >= POINT_THRESH:
                point_x_1 = round(kp[line_points[i][0] * 3] * image.shape[1])
                point_y_1 = round(kp[line_points[i][0] * 3 + 1] * image.shape[0])
                point_x_2 = round(kp[line_points[i][1] * 3] * image.shape[1])
                point_y_2 = round(kp[line_points[i][1] * 3 + 1] * image.shape[0])
                cv.line(image, (point_x_1, point_y_1), (point_x_2, point_y_2), colors[line_colors[i]], 2)


if __name__ == '__main__':

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

    yolov3 = KSNN('VIM3')
    print(' |---+ KSNN Version: {} +---| '.format(yolov3.get_nn_version()))

    print('Start init neural network ...')
    yolov3.nn_init(library=library, model=model, level=level)
    print('Done.')

    print('Get input data ...')
    cv_img =  list()
    orig_img = cv.imread(picture, cv.IMREAD_COLOR)
    img = cv.resize(orig_img, (640, 640)).astype(np.float32)
    img[:, :, 0] = img[:, :, 0] - mean[0]
    img[:, :, 1] = img[:, :, 1] - mean[1]
    img[:, :, 2] = img[:, :, 2] - mean[2]
    img = img / var[0]
    
    img = img.transpose(2, 0, 1)
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv_img.append(img)
    print('Done.')

    print('Start inference ...')
    start = time.time()

    '''
        default input_tensor is 1
    '''
    data = yolov3.nn_inference(cv_img, platform='ONNX', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)
    end = time.time()
    print('Done. inference time: ', end - start)

    input0_data = data[2]
    input1_data = data[1]
    input2_data = data[0]

    input0_data = input0_data.reshape(SPAN, LISTSIZE, GRID0, GRID0)
    input1_data = input1_data.reshape(SPAN, LISTSIZE, GRID1, GRID1)
    input2_data = input2_data.reshape(SPAN, LISTSIZE, GRID2, GRID2)

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
    
    boxes, scores, classes, key_points = yolov3_post_process(input_data)

    if boxes is not None:
        draw(orig_img, boxes, scores, classes, key_points)

    cv.imwrite("./result.jpg", orig_img)
    cv.imshow("results", orig_img)
    cv.waitKey(0)
