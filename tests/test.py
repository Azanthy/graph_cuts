#! /usr/bin/env python3

import sys, os
from skimage.io import imread
from sklearn.metrics import f1_score
import numpy as np

PATH_TO_DATASET = "../segmentation_dataset/"
BUILD_FOLDER = "../build/"
NAME_OF_EXEC = "graph_cuts"
NUMBER_OF_TESTS = 4 # -1 to test the whole dataset

if __name__ == '__main__':
    files = os.listdir(os.path.join(PATH_TO_DATASET, "inputs", "normal"))
    if NUMBER_OF_TESTS != -1:
        files = files[:NUMBER_OF_TESTS]
    exec_path = os.path.join(BUILD_FOLDER, NAME_OF_EXEC)
    scores = []
    for target_file in files:
        print("=======================================")
        print("Creating " + target_file)
        os.system(exec_path + ' ' + 
            os.path.join(PATH_TO_DATASET, "inputs", "normal", target_file) + ' ' +
            os.path.join(PATH_TO_DATASET, "inputs", "marked", "marked_" + target_file))
        if os.path.exists("output.jpg"):
            print("Created " + target_file + " output")
        else:
            print("Unable to create " + target_file + " output")
            continue
        os.rename("output.jpg", "output_" + target_file)
        output = imread("output_" + target_file, as_gray=True).astype('int') * 255
        expected = imread(os.path.join(PATH_TO_DATASET, "expected_outputs", target_file), as_gray=True).astype('int')
        print(output.max(), expected.max())
        score = f1_score(expected.reshape(-1), output.reshape(-1), average='micro')
        scores.append(score)
        print("DICE_COEFFICIENT : " + str(score))
    print("================")
    print("Mean DICE " + str(np.array(scores).mean()) + " on " + str(len(scores)) + " tests.")