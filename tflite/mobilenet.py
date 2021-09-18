import numpy as np
import os      
import argparse
import sys
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import time

def show_top5(outputs):
	output = outputs[0].reshape(-1)
	output_sorted = sorted(output, reverse=True)
	top5_str = '----Mobilenet----\n-----TOP 5-----\n'
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

	mobilenet_tflite = KSNN('VIM3')
	print(' |---+ KSNN Version: {} +---| '.format(mobilenet_tflite.get_nn_version()))

	print('Start init neural network ...')
	mobilenet_tflite.nn_init(c_lib_p = solib, nb_p = nbfile)
	print('Done.')

	print('Get input data ...')
	img = cv.imread( args.input_picture, cv.IMREAD_COLOR )
	print('Done.')

	print('Start inference ...')
	start = time.time()
	outputs = mobilenet_tflite.nn_inference(img, platform = 'TFLITE', reorder='0 1 2', out_format=out_format.OUT_FORMAT_FLOAT32)
	end = time.time()
	print('Done. inference : ', end - start)

	show_top5(outputs)


