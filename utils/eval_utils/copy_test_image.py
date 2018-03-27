from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import time
import cv2
import shutil
import random

def copy_test_image(imagesetfile, src_path, dst_path, shuffle = False, visual_num = 100):
    with open(imagesetfile) as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    print("src image path: ", os.path.abspath(src_path))
    print("dst image path: ", os.path.abspath(dst_path))

    idx = 0
    if shuffle:
        random.shuffle(splitlines)

    draw_image_list = []

    for line in splitlines:
        idx += 1
        src_file = os.path.join(src_path, line[0] + '.jpg')
        shutil.copy(src_file, dst_path)
        draw_image_list.append(line[0])
        if idx >= visual_num:
            break
    
    return draw_image_list

# if __name__ == '__main__':
    # copy_test_image('/home/jwwangchn/data/VOCdevkit/VOC220/ImageSets/Main/test.txt', None, None)