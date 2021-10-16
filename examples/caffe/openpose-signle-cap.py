import numpy as np
import os
import argparse
import sys
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import time

nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
threshold = 0.1

if __name__ == "__main__":

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
    else :
        sys.exit("NBG file not found !!! Please use format: --model")
    if args.device :
        cap_num = args.device
    else :
        sys.exit("video device not found !!! Please use format :--video-device ")

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

    np.set_printoptions(threshold=np.inf)

    openpose = KSNN('VIM3')
    print(' |---+ KSNN Version: {} +---| '.format(openpose.get_nn_version()))
    print('Start init neural network ...')
    openpose.nn_init(library=library, model=model, level=level)
    print('Done.')

    Keypoint = 'Output-Keypoints'
    cv.namedWindow(Keypoint)

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
            default output_tensor is 1
        '''
        outputs = openpose.nn_inference(cv_img, platform = 'CAFFE', reorder = '2 1 0', output_format=output_format.OUT_FORMAT_FLOAT32)
        end = time.time()
        print('Inference time: ', end - start)
        output = outputs[0].reshape(1, 57, 46, 46)

        H = output.shape[2]
        W = output.shape[3]
    
        points = []

        for i in range(nPoints):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv.minMaxLoc(probMap)
            x = (img.shape[1] * point[0]) / W
            y = (img.shape[0] * point[1]) / H

            if prob > threshold :
                print(prob)
                cv.circle(img, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
                cv.putText(img, "{}".format(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv.LINE_AA)
                points.append((int(x), int(y)))
            else:
                points.append(None)

        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                cv.line(img, points[partA], points[partB], (0, 255, 255), 2)
                cv.circle(img, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv.FILLED)

        cv.imshow(Keypoint, img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()





