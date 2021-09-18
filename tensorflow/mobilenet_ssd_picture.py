from ctypes import *
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

	cv.imwrite("out.jpg", img)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--nb-file", help="path to nb file")
	parser.add_argument("--so-lib", help="path to so lib")
	parser.add_argument("--input-picture", help="path to input picture")
	args = parser.parse_args()
	if args.nb_file :
		if os.path.exists(args.nb_file) == False:
			sys.exit('nb file \'' + args.nb_file + '\' not exist')
		nbfile = args.nb_file
	else :
		sys.exit("nb file not found !!! Please specify argument: --nb-file /path/to/nb-file")
	if args.input_picture :
		if os.path.exists(args.input_picture) == False:
			sys.exit('input picture \'' + args.input_picture + '\' not exist')
		inputpicturepath = bytes(args.input_picture,encoding='utf-8')
	else :
		sys.exit("input picture not found !!! Please specify argument: --input-picture /path/to/picture")
	if args.so_lib :
		if os.path.exists(args.so_lib) == False:
			sys.exit('so lib \'' + args.so_lib + '\' not exist')
		solib = args.so_lib
	else :
		sys.exit("so lib not found !!! Please specify argument: --so-lib /path/to/lib")

	ssd = KSNN('VIM3')
	print(' |---+ KSNN Version: {} +---| '.format(ssd.get_nn_version()))
	ssd.nn_init(c_lib_p = solib, nb_p = nbfile)
	img = cv.imread( args.input_picture, cv.IMREAD_COLOR )

	start = time.time()
	outputs = ssd.nn_inference(img,platform = 'TENSORFLOW', num=2, reorder='0 1 2', out_format = out_format.OUT_FORMAT_FLOAT32)
	end = time.time()
	print('inference : ', end - start)

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

		if topClassScore > 0.4:
			candidateBox[0][vaildCnt] = i
			candidateBox[1][vaildCnt] = topClassScoreIndex
			scoreBox[0][vaildCnt] = topClassScore
			vaildCnt += 1

	# calc position
	calc_position(vaildCnt, candidateBox, predictions, box_priors)

	# NMS
	nms(vaildCnt, candidateBox, predictions)

	# Draw result
	draw(img, vaildCnt, candidateBox, predictions, scoreBox)

