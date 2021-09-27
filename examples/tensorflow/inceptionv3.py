import numpy as np
import os      
import argparse
import sys
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import time

def show_top5( f32_data ):
    list_result = []
    for index in range( len(f32_data) ):
        list_result.append((f32_data[index], index))
    list_result.sort(reverse=True)
    print("----- Show Top5 +-----")
    for i in range(5):
        print("{:>6d}: {:.5f}".format(list_result[i][1],list_result[i][0]))


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

    inception = KSNN('VIM3')
    print(' |--- KSNN Version: {} +---| '.format(inception.get_nn_version()))

    print('Start init neural network ...')
    inception.nn_init(library=library, model=model, level=level)
    print('Done.')

    print('Get input data ...')
    cv_img = list()
    img = cv.imread(picture, cv.IMREAD_COLOR)
    cv_img.append(img)
    print('Done.')

    print('Start inference ...')
    start = time.time()
    '''
        default input_tensor is 1
        default output_tensor is 1
    '''
    outputs = inception.nn_inference(cv_img, platform = 'TENSORFLOW', reorder='0 1 2', output_format=output_format.OUT_FORMAT_FLOAT32)
    end = time.time()
    print('Done. inference : ', end - start)


    show_top5(outputs[0])


