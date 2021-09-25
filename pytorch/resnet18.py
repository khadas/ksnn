import numpy as np
import os      
import argparse
import sys
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import time

def show_top5(output):
	output_sorted = sorted(output, reverse=True)
	top5_str = '----Resnet18----\n-----TOP 5-----\n'
	for i in range(5):
		value = output_sorted[i]
		index = np.where(output == value)
		for j in range(len(index)):
			if (i + j) >= 5:
				break
			if value > 0:
				topi = '{}: {}\n'.format(index[j], value)
			else:
				topi = '-1: 0.0\n'
			top5_str += topi
	print(top5_str)

def softmax(x):
	return np.exp(x)/sum(np.exp(x))

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

	resnet18 = KSNN('VIM3')
	print(' |---+ KSNN Version: {} +---| '.format(resnet18.get_nn_version()))

	print('Start init neural network ...')

	resnet18.nn_init(c_lib_p = solib, nb_p = nbfile, level=0)

	print('Done.')

	print('Get input data ...')
	cv_img = []
	img = cv.imread( args.input_picture, cv.IMREAD_COLOR )
	cv_img.append(img)
	print('Done.')

	print('Start inference ...')
	start = time.time()
	outputs = resnet18.nn_inference(cv_img, platform = 'PYTORCH', reorder='2 1 0', out_format=out_format.OUT_FORMAT_FLOAT32)
	end = time.time()
	print('Done. inference : ', end - start)

	show_top5(softmax(np.array(outputs[0],dtype=np.float32)))


