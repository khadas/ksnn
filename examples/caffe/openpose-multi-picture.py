import numpy as np
import os
import argparse
import sys
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import time
from random import randint

nPoints = 18

model_width = 368
model_height = 368

keypoints_mapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,16], [5,17] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
map_ids = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


threshold = 0.03

def get_keypoints(probmap, threshold=0.03):

    mapSmooth = cv.GaussianBlur(probmap,(3,3),0,0)

    mapmask = np.uint8(mapSmooth>threshold)
#    np.set_printoptions(threshold=np.inf)
#    print(mapmask)
    keypoints = []

    #find the blobs
    contours, hierarchy = cv.findContours(mapmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        blobmask = np.zeros(mapmask.shape)
        blobmask = cv.fillConvexPoly(blobmask, cnt, 1)
        maskedprobmap = mapSmooth * blobmask
        _, maxval, _, maxloc = cv.minMaxLoc(maskedprobmap)
        keypoints.append(maxloc + (probmap[maxloc[1], maxloc[0]],))
    return keypoints


# Find valid connections between the different joints of a all persons present
def get_valid_pairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 5
    paf_score_th = 0.03
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(map_ids)):
        # A->B constitute a limb
        pafA = output[0, map_ids[k][0], :, :]
        pafB = output[0, map_ids[k][1], :, :]
        pafA = cv.resize(pafA, (model_width, model_height))
        pafB = cv.resize(pafB, (model_width, model_height))

        # candA: (124, 365, 0.17102814, 43)
        #                               detected_keypoints keypoint_id
        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]

        nA = len(candA)
        nB = len(candB)
#        print(nA,nB)
        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                max_score = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > max_score:
                            max_j = j
                            max_score = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    #   detected_keypoints keypoint_id
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], max_score]], axis=0)
            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score

    personwiseKeypoints = -1 * np.ones((0, 19))
    for k in range(len(map_ids)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
#    print(personwiseKeypoints)
    return personwiseKeypoints




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--library", help="Path to C static library file")
    parser.add_argument("--model", help="Path to nbg file")
    parser.add_argument("--level", help="Information printer level: 0/1/2")
    parser.add_argument("--picture", help="Path to input picture")

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

    np.set_printoptions(threshold=np.inf)

    openpose = KSNN('VIM3')
    print(' |---+ KSNN Version: {} +---| '.format(openpose.get_nn_version()))
    print('Start init neural network ...')
    openpose.nn_init(library=library, model=model, level=level)
    print('Done.')

    Keypoint = 'Output-Keypoints'
    cv.namedWindow(Keypoint)

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
    outputs = openpose.nn_inference(cv_img, platform = 'CAFFE', reorder = '2 1 0', output_format=output_format.OUT_FORMAT_FLOAT32)
    end = time.time()
    print('Done. inference time: ', end - start)
    output = outputs[0].reshape(1, 57, 46, 46)

    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0

    for part in range(nPoints):
        probmap = output[0, part, :, :]
        probmap = cv.resize(probmap, (model_width, model_height))
        keypoints = get_keypoints(probmap, threshold)
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

    valid_pairs, invalid_pairs = get_valid_pairs(output)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

    for i in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0]*img.shape[1]/model_width)
            A = np.int32(keypoints_list[index.astype(int), 1]*img.shape[0]/model_height)
            cv.line(img, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv.LINE_AA)


    cv.imshow(Keypoint, img)
    cv.waitKey(0)
    cv.destroyAllWindows()





