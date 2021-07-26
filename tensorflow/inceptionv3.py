from ctypes import *
import numpy as np
import os      
import argparse
import sys
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv

def show_top5( f32_data ):
	list_result = []
	for index in range( len(f32_data) ):
		list_result.append((f32_data[index], index))
	list_result.sort(reverse=True)
	print("-----+ Show Top5 +-----")
	for i in range(5):
		print("{:>6d}: {:.5f}".format(list_result[i][1],list_result[i][0]))


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

	inception = KSNN('VIM3')
	inception.nn_version()
	inception.nn_init(c_lib_p = solib, nb_p = nbfile)
	img = cv.imread( args.input_picture, cv.IMREAD_COLOR )
	f32_data = inception.nn_inference(img,platform = 'TENSORFLOW',out_format = out_format.OUT_FORMAT_FLOAT32)
	show_top5(f32_data[0])


