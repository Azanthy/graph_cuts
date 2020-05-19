#! /usr/bin/env python3

import cv2, sys, os
import numpy as np

def main(argv):
    if len(argv) != 1:
        return
    binary = argv[0]
    INPUT_PATH = '../segmentation_dataset/inputs/'
    EXPECTED_PATH = '../segmentation_dataset/expected_outputs/'
    imgs_normal = [ f for f in os.listdir(INPUT_PATH + 'normal/') ]
    imgs_marked = [ "marked_" + f for f in imgs_normal ]
    print([p[:-4] for p in imgs_normal])
    for i in range(len(imgs_normal)):
        line = binary + " -i " + INPUT_PATH+'normal/'+imgs_normal[i] + " " + \
                                 INPUT_PATH+'marked/'+imgs_marked[i] + \
                        " -o " + "out/"+imgs_normal[i]
        print(line) #replace here by the execution of the command line
        expected_name = imgs_normal[i]
        if expected_name[-3:] == "jpg":
            expected_name = expected_name[:-3] + "png"
        expected = cv2.imread(os.path.abspath(EXPECTED_PATH+expected_name))
        got = cv2.imread(os.path.abspath("out/"+expected_name))
        diff = np.count_nonzero(cv2.bitwise_xor(expected, got))
        width, height, _ = expected.shape
        percentage = 100. * float(diff) / float(width * height)
        print("Percentage of difference for "+imgs_normal[i]+ " is "+ str(percentage))

if __name__ == "__main__":
    main(sys.argv[1:])
